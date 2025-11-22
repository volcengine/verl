# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
from copy import deepcopy
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from ray.util.queue import Queue
from tensordict import TensorDict

from recipe.rollpacker.queue_util import QueueMonitor
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    _agent_loop_registry,
    _DummyConfig,
    _InternalAgentLoopOutput,
    get_trajectory_info,
)
from verl.protocol import DataProto, DataProtoItem
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import (
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncReorderLLMServerManager(AsyncLLMServerManager):
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        super().__init__(config, server_handles, max_cache_size)
        self.ray_tasks = []

    def cancel_tasks(self):
        for ray_task in self.ray_tasks:
            ray.cancel(ray_task)
        self.ray_tasks = []

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        try:
            server = self._choose_server(request_id)
            task = server.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )
            self.ray_tasks.append(task)
            output = await task
            return output
        except Exception as e:
            logger.error(f"server manager got exception: {e}")


def _postprocess(inputs: list[_InternalAgentLoopOutput]) -> DataProto:
    """Process the padded outputs from _run_agent_loop and combine them into a batch."""
    # Convert lists back to tensors and stack them to create a batch.
    prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
    response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
    response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
    attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
    input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
    position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
    optional_outputs = {}
    if inputs[0].response_logprobs is not None:
        optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

    batch = TensorDict(
        {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            "responses": response_ids,  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
            "position_ids": position_ids,
            **optional_outputs,
        },
        batch_size=len(inputs),
    )

    scores = [input.reward_score for input in inputs]
    if all(score is not None for score in scores):
        prompt_length = prompt_ids.size(1)
        response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
        batch["rm_scores"] = rm_scores

    non_tensor_batch = {
        "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
    }

    # add reward_extra_info to non_tensor_batch
    reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
    reward_extra_keys = list(reward_extra_infos[0].keys())
    for key in reward_extra_keys:
        non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

    # Add multi_modal_inputs to non_tensor_batch if any samples have them
    multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
    if any(mmi is not None for mmi in multi_modal_inputs_list):
        non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

    metrics = [input.metrics.model_dump() for input in inputs]
    # Collect extra fields from all inputs and convert them to np.ndarray
    extra_fields = {}
    all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
    for key in all_keys:
        temp_arr = np.empty(len(inputs), dtype=object)
        temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
        extra_fields[key] = temp_arr

    non_tensor_batch.update(extra_fields)
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
    )


@ray.remote
class AgentLoopReorderWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        self.server_manager = AsyncReorderLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)
        self.queue = None
        self.unfinished_queue = None
        self.tasks = []

    def set_queue(self, queue: QueueMonitor):
        self.queue = queue

    def set_unfinished_queue(self, unfinished_queue: Queue):
        self.unfinished_queue = unfinished_queue

    @tqbridge()
    async def generate_sequences(self, batch: DataProto):
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        raw_batch = deepcopy(batch)

        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        self.tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            self.tasks.append(
                asyncio.create_task(
                    self._run_reorder_agent_loop(sampling_params, trajectory_info[i], raw_batch[i], **kwargs)
                )
            )

        results = await asyncio.gather(*self.tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                await self.unfinished_queue.put_async(raw_batch[i])

    async def _run_reorder_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        raw_item: DataProtoItem,
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = getattr(output, "multi_modal_data", {}).get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            enable_async_reward = (
                self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            await self.queue.put.remote(
                (
                    raw_item,
                    _InternalAgentLoopOutput(
                        prompt_ids=prompt_output["input_ids"],
                        response_ids=response_output["input_ids"],
                        input_ids=input_ids,
                        position_ids=position_ids,
                        response_mask=response_mask,
                        attention_mask=attention_mask,
                        response_logprobs=response_logprobs,
                        multi_modal_inputs=multi_modal_inputs,
                        multi_modal_data=output.multi_modal_data,
                        reward_score=output.reward_score,
                        num_turns=output.num_turns,
                        metrics=output.metrics,
                        extra_fields=output.extra_fields,
                    ),
                )
            )

    async def cancel_tasks(self):
        for task in self.tasks:
            task.cancel()
        self.server_manager.cancel_tasks()


class ReorderAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        self.agent_loop_workers_class = AgentLoopReorderWorker
        super().__init__(config, worker_group, rm_wg)
        self.queue = QueueMonitor.remote(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n)
        for worker in self.agent_loop_workers:
            ray.get(worker.set_queue.remote(self.queue))
            ray.get(self.queue.append_worker.remote(worker))

    def get_threshold(self):
        return ray.get(self.queue.get_threshold.remote())

    def set_threshold(self, threshold: int):
        ray.get(self.queue.set_threshold.remote(threshold))

    def set_unfinished_queue(self, unfinished_queue):
        for worker in self.agent_loop_workers:
            ray.get(worker.set_unfinished_queue.remote(unfinished_queue))

    def generate_sequences(self, prompts: DataProto) -> tuple[DataProto, list[Any]]:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        tasks = [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
        ]
        try:
            ray.get(tasks)
        except ray.exceptions.TaskCancelledError as e:
            logger.warning(f"Task cancelled: {e}")

        hybird_out = ray.get(self.queue.clear.remote())

        tasks_outputs = []
        raw_items = []
        for raw_item, tasks_output in hybird_out:
            tasks_outputs.append(tasks_output)
            raw_items.append(raw_item)

        batch_output = _postprocess(tasks_outputs)

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [batch_output.meta_info.pop("metrics")]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, batch_output)

        batch_output.meta_info = {"timing": timing, **batch_output.meta_info}
        return batch_output, raw_items
