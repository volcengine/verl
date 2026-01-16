# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import time
from typing import Any, Optional, Sequence

import hydra
import numpy as np
import ray
from omegaconf import DictConfig

from recipe.partial_rollout.vllm_rollout.vllm_async_server import PRv3vLLMReplica
from recipe.partial_rollout.prompt_manager import RolloutPromptManager, RolloutPrompt
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    DictConfigWrap,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.rollout_trace import (
    rollout_trace_attr,
    rollout_trace_op,
)

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


class PRv3AsyncLLMServerManager(AsyncLLMServerManager):
    @rollout_trace_op
    async def generate_for_partial(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> tuple[list[Any], list[Any], Any] | tuple[Sequence[int], list[float], bool]:
        """Generate tokens from prompt ids, used for async partial.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            output: A tuple representing the generation output.
            - Element 0 (Sequence[int]): Generated response token IDs.
            - Element 1 (list[float]): Log probabilities for the response token IDs.
            - Element 2 (bool): A flag or status indicating cancellation.
        """
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output



@ray.remote
class PRv3AgentLoopWorker(AgentLoopWorkerBase):
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None, prompt_manager_handler: ray.actor.ActorHandle = None):
        self.server_manager = PRv3AsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)
        self.cancellation_event = asyncio.Event()
        self.prompt_manager_handler = prompt_manager_handler
    
    async def generate_sequences_async(self, batch: DataProto) -> bool:
        num_rollout_prompts = batch.batch.size(0) // self.config.actor_rollout_ref.rollout.n
        num_rollout_prompts = int(num_rollout_prompts * 1)
        rollout_prompts: list[RolloutPrompt] = ray.get(
            self.prompt_manager_handler.pull_pending_prompts.remote(num_rollout_prompts)
        )

        running_set: set[asyncio.Task] = {
            asyncio.create_task(self._generate_sequences_no_post(rp))
            for rp in rollout_prompts
        }

        while running_set:
            done, _ = await asyncio.wait(running_set, return_when=asyncio.FIRST_COMPLETED)
            logger.info(f"[PRv3AgentLoopWorker] done: {len(done)}")
            for task in done:
                running_set.remove(task)
                
                rollout_prompt, is_cancel = task.result()
                logger.info(f"[PRv3AgentLoopWorker] generate_sequences_async: is_cancel: {is_cancel}")
                ray.get(
                    self.prompt_manager_handler.push_done_prompt.remote(rollout_prompt, is_cancel)
                )
                logger.info(f"[PRv3AgentLoopWorker] push_done_prompt done")

                if self.cancellation_event.is_set():
                    continue

                new_rollout_prompts: list[RolloutPrompt] = ray.get(
                    self.prompt_manager_handler.pull_pending_prompts.remote(1)
                )
                
                running_set.update(
                    asyncio.create_task(self._generate_sequences_no_post(new_rp))
                    for new_rp in new_rollout_prompts
                )
        return "DONE"

    async def _generate_sequences_no_post(self, rollout_prompt: RolloutPrompt,) -> tuple[RolloutPrompt, bool]:
        """Generate sequences from agent loop. (one rollout prompt with n rollout samples)

        Args:
            rollout_prompt (RolloutPrompt): Rollout prompt (one prompt with n rollout samples).

        Returns:
            list[AgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        # batch (DataProto): Input batch (one prompt with n rollout samples).
        # partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.
        batch = rollout_prompt.full_batch
        partial_output_list = rollout_prompt.agent_loop_output_list
        config = self.config.actor_rollout_ref.rollout
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
        
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)
        try:
            tasks = []
            for i in range(len(batch)):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                kwargs["output"] = partial_output_list[i]
                tasks.append(
                    asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
                )
            outputs = await asyncio.gather(*tasks)
        except Exception:
            logger.exception("_partial_run_agent_loop failed")
            raise

        is_cancel = any(output.extra_fields.get("is_cancel", False) for output in outputs)
        if not is_cancel:
            output = self._postprocess(outputs)
            output = self._addition_process(output)
            rollout_prompt.full_batch = output
            rollout_prompt.agent_loop_output_list = []
            return rollout_prompt, is_cancel
        else:
            rollout_prompt.agent_loop_output_list = outputs
            return rollout_prompt, is_cancel

    def _addition_process(self, output: DataProto):
        """collect metirics"""
        metrics = output.meta_info["metrics"]  # List[Dict[str, str]]
        processing_times_list = [item["generate_sequences"] for item in metrics]
        tool_calls_times_list = [item["tool_calls"] for item in metrics]
        output.non_tensor_batch["processing_times"] = processing_times_list
        output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
        return output

    async def _partial_run_agent_loop(
        self, sampling_params: dict[str, Any], trajectory: dict[str, Any], *, agent_name: str, **kwargs,
    ) -> AgentLoopOutput:
        """Run agent loop for partial rollout (one sample within a prompt)"""
        # Completed, return directly
        if kwargs["output"] is not None and not kwargs["output"].extra_fields.get("is_cancel", False):
            logger.info("In _partial_run_agent_loop, already completed, return derictly!")
            return kwargs["output"]
        try:
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
                    trainer_config=DictConfigWrap(config=self.config),
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                )
                output: AgentLoopOutput = await agent_loop.run(
                    sampling_params, cancellation_event=self.cancellation_event, **kwargs
                )
                if not output.extra_fields.get("is_cancel", False):
                    kwargs.pop("output", None)
                    output = await self._agent_loop_postprocess(output, **kwargs)
                return output
        except Exception:
            logger.exception("Agent_loop run failed")
            raise

    async def cancel_agent_loops(self):
        """Set the shared cancellation event to stop all agent loops."""
        self.cancellation_event.set()

    async def resume_agent_loops(self):
        """Clear the shared cancellation event."""
        self.cancellation_event.clear()




class PRv3AgentLoopManager(AgentLoopManager):
    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, 
        rm_resource_pool: RayResourcePool = None, prompt_manager_handler: ray.actor.ActorHandle = None,
    ):
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            # TODO (dyy): current rm is colocated with the legacy fsdp/megatron rm
            # future pr will depericate fsdp/megatron rm and init RewardModelManager in standalone mode
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        self.rollout_replica_class = PRv3vLLMReplica
        self.agent_loop_workers_class = PRv3AgentLoopWorker
        self.prompt_manager_handler = prompt_manager_handler

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address, self.prompt_manager_handler)
            )
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers."""
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        if prompts.meta_info.get("validate", False):
            self.resume()
            outputs = ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
                ]
            )
            # In sync rollout mode, no need to call cancel()
        else:
            self.resume()
            # 1. Prepare generation
            num_rollout_prompts = prompts.batch.size(0) // self.config.actor_rollout_ref.rollout.n
            ray.get(self.prompt_manager_handler.prepare_generation.remote(prompts.meta_info.get("global_steps", 0)))
            # 2. Launch all AgentLoopWorker's generate_sequences_async task
            worker_tasks = [
                worker.generate_sequences_async.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
            # 3. Monitor generation (if cache containing `num_rollout_prompts` or dataloader is exhausted, return)
            while True:
                done = ray.get(self.prompt_manager_handler.check_generation_once.remote(num_rollout_prompts))
                if done:
                    logger.info(f"[PRv3AgentLoopManager] check_generation_once done: {done}")
                    break
                time.sleep(0.01)
            # 4. Cancel all AgentLoopWorker's generate_sequences_async task
            self.cancel()
            # 5. Wait for all AgentLoopWorker's generate_sequences_async task to return "DONE"
            assert all(result == "DONE" for result in ray.get(worker_tasks)), "PRv3AgentLoopWorker generate sequences failed"
            # 6. Pull valid prompts from prompt manager
            is_full = ray.get(self.prompt_manager_handler.check_generation_post_state.remote(num_rollout_prompts))
            outputs = ray.get(self.prompt_manager_handler.pull_done_prompts.remote(num_rollout_prompts))
            outputs[0].meta_info["is_full"] = is_full
        
        output = DataProto.concat(outputs)

        # Fix for Issue #4147: Always call sleep() to ensure proper cleanup
        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        if output.meta_info.get("is_full", True):
            metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
            timing = self._performance_metrics(metrics, output)

            output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def cancel(self):
        worker_cancel_tasks = [worker.cancel_agent_loops.remote() for worker in self.agent_loop_workers]
        ray.get(worker_cancel_tasks)
        rollout_cancel_tasks = [replica.cancel() for replica in self.rollout_replicas]
        self._run_all(rollout_cancel_tasks)

    def resume(self):
        rollout_resume_tasks = [replica.resume() for replica in self.rollout_replicas]
        self._run_all(rollout_resume_tasks)
        worker_resume_tasks = [worker.resume_agent_loops.remote() for worker in self.agent_loop_workers]
        ray.get(worker_resume_tasks)


