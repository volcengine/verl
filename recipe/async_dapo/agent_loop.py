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
import heapq
import logging
import os
import random
import threading
from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from pydantic import BaseModel
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.profiler import simple_timer
from verl.utils.reward_score.math_dapo import compute_score
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class EarlyStoppingCoordinator:
    """Global state manager for coordinating early stopping mechanism"""

    def __init__(self, expected_prompt_num: int):
        self.expected_prompt_num = expected_prompt_num
        self.completed_prompts = set()  # Set of completed prompt sample_indices
        self.invalid_prompt_count = 0  # Count of invalid prompts
        self.should_stop = False
        self.lock = threading.Lock()
        print(f"[EarlyStoppingCoordinator] Initialized: expected_prompt_num={expected_prompt_num}")

    def report_completion(self, sample_index: int, is_valid: bool) -> bool:
        """Report completion status of a prompt

        Args:
            sample_index: sample_index of the completed prompt
            is_valid: whether it's valid
        Returns:
            bool: whether early stopping should be triggered
        """
        with self.lock:
            if self.should_stop:
                return True

            if is_valid:
                self.completed_prompts.add(sample_index)
            else:
                self.invalid_prompt_count += 1

            completed_count = len(self.completed_prompts)

            if completed_count >= self.expected_prompt_num:
                self.should_stop = True
                print(
                    f"[EarlyStoppingCoordinator] Early stopping triggered: "
                    f"{completed_count}/{self.expected_prompt_num} prompts completed, "
                    f"invalid prompt count: {self.invalid_prompt_count}"
                )
                return True
            else:
                print(
                    f"[EarlyStoppingCoordinator] Progress update: "
                    f"{completed_count}/{self.expected_prompt_num} prompts completed, "
                    f"invalid prompt count: {self.invalid_prompt_count}"
                )
                return False

    def should_stop_generation(self) -> bool:
        """Check whether generation should stop"""
        with self.lock:
            return self.should_stop

    def get_completed_prompts(self) -> set:
        """Get the set of completed prompts"""
        with self.lock:
            return self.completed_prompts.copy()


@ray.remote(concurrency_groups={"acquire": 1, "release": 10, "reset": 1})
class GlobalLoadBalancer:
    """
    Global load balancer that only handles server index allocation, not actual generate calls
    Uses threading.Semaphore instead of asyncio.Queue to avoid concurrency issues in Ray Actor
    """

    def __init__(self, config: DictConfig, num_servers: int, max_cache_size: int = 10000):
        """Initialize the GlobalLoadBalancer.

        Args:
            config (DictConfig): YAML config.
            num_servers (int): Number of servers
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.num_servers = num_servers

        # TODO: add a config for this
        self.max_loads_per_server = 256
        self.total_capacity = self.max_loads_per_server * num_servers
        self._semaphore = threading.Semaphore(self.total_capacity)
        self._current_loads = [0] * num_servers  # Track current load of each server
        self._lock = threading.Lock()  # Protect concurrent access to _current_loads

        print(f"[GlobalLoadBalancer] max_loads_per_server: {self.max_loads_per_server}")
        print(f"[GlobalLoadBalancer] total_capacity: {self.total_capacity}")

        # Least requests load balancing
        self.weighted_serveres = [[0, server_index] for server_index in range(num_servers)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    @ray.method(concurrency_group="acquire")
    def get_server_index(self, request_id: str) -> int:
        """Get the server index that should be used"""
        if self.config.actor_rollout_ref.rollout.get("load_balance", False):
            # Acquire semaphore permission
            self._semaphore.acquire()

            # Select server with minimum load
            with self._lock:
                min_load_idx = min(range(self.num_servers), key=lambda i: self._current_loads[i])
                self._current_loads[min_load_idx] += 1
                server_index = min_load_idx

            return server_index
        else:
            return self._choose_server_index(request_id)

    @ray.method(concurrency_group="release")
    def release_server_index(self, server_index: int):
        """Release server index"""
        if self.config.actor_rollout_ref.rollout.get("load_balance", False):
            # Decrease server load count
            with self._lock:
                if self._current_loads[server_index] > 0:
                    self._current_loads[server_index] -= 1

            # Release semaphore permission
            self._semaphore.release()

    def _choose_server_index(self, request_id: str) -> int:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server_index = self.weighted_serveres[0][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server_index
        return server_index

    @ray.method(concurrency_group="reset")
    def reset(self):
        """Reset load balancer state, including semaphore and load counts"""
        with self._lock:
            # Recreate semaphore
            self._semaphore = threading.Semaphore(self.total_capacity)
            # Reset load counts for all servers
            self._current_loads = [0] * self.num_servers

        print("[GlobalLoadBalancer] Load balancer state reset")


class AsyncLLMServerManager:
    """
    Local server manager responsible for actual generate calls
    Gets server allocation through global load balancer
    """

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        global_load_balancer: ray.actor.ActorHandle,
    ):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            global_load_balancer (ray.actor.ActorHandle): Handle to global load balancer
        """
        self.config = config
        self.server_handles = server_handles
        self.global_load_balancer = global_load_balancer

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        # Get server index from global load balancer (now synchronous call)
        server_index = await self.global_load_balancer.get_server_index.remote(request_id)
        server = self.server_handles[server_index]
        output = None

        try:
            output = await server.generate_with_cancel.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        except asyncio.CancelledError:
            print(f"[AsyncLLMServerManager] Task cancelled: {request_id}")
            await server.cancel.remote(request_id)
        finally:
            # Ensure server index is released even if exception occurs (now synchronous call)
            await self.global_load_balancer.release_server_index.remote(server_index)

        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class RewardOutput(BaseModel):
    """Reward output."""

    reward: float = 0.0
    acc: float = 0.0
    pred: str = ""


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    rollout_index: int = -1
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int = 0
    metrics: AgentLoopMetrics
    reward: RewardOutput = RewardOutput()


# the config API has been changed, so we need to use the old API
# class SingleTurnAgentLoop(AgentLoopBase):
class SingleTurnAgentLoop:
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, config, server_manager, tokenizer):
        # super().__init__(config, server_manager, tokenizer)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()

    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )

        with simple_timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        if response_ids is None:
            return None
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
        )
        return output


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        global_load_balancer: ray.actor.ActorHandle,
        server_handles: list[ray.actor.ActorHandle],
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            global_load_balancer (ray.actor.ActorHandle): Handle to global load balancer
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        # Create local server manager using global load balancer
        self.server_manager = AsyncLLMServerManager(config, server_handles, global_load_balancer)
        self.early_stopping_coordinator = None  # Early stopping coordinator, set during generate_sequences
        self.max_concurrent_prompts = config.actor_rollout_ref.rollout.get("max_concurrent_prompts", 32)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        trace_config = config.trainer.get("rollout_trace", {})

        RolloutTraceConfig.init(
            config.trainer.project_name,
            config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(
        self, batch: DataProto, early_stopping_coordinator: ray.actor.ActorHandle = None
    ) -> DataProto:
        """Generate sequences from agent loop with dynamic task creation and early stopping support.

        Args:
            batch (DataProto): Input batch.
            early_stopping_coordinator: Early stopping coordinator

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
        self.early_stopping_coordinator = early_stopping_coordinator
        is_validation = batch.meta_info.get("validate", False)

        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))
        rollout_index = batch.non_tensor_batch["rollout_index"]
        ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in batch]

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index, rollout_index)

        # Group tasks by prompt for early stopping management
        prompt_groups = self._group_by_prompt(agent_names, raw_prompts, ground_truths, trajectory_info, sampling_params)

        # Dynamic task creation and management
        completed_outputs = {}
        pending_tasks = {}  # task -> sample_index mapping

        # Create pending prompt queue (using list to maintain order)
        pending_prompts = list(prompt_groups.items())

        # Set maximum concurrent tasks (configurable)
        max_concurrent_tasks = self.max_concurrent_prompts
        print(f"[AgentLoopWorker] Dynamic task creation mode, max concurrent tasks: {max_concurrent_tasks}")
        print(f"[AgentLoopWorker] Total {len(pending_prompts)} prompt groups to process")

        # Initially create a batch of tasks
        created_task_count = 0
        for _ in range(min(max_concurrent_tasks, len(pending_prompts))):
            if pending_prompts:
                sample_index, group_data = pending_prompts.pop(0)
                task = asyncio.create_task(
                    self._run_prompt_group(sample_index, group_data, do_filter=not is_validation)
                )
                pending_tasks[task] = sample_index
                created_task_count += 1
                print(f"[AgentLoopWorker] Initially created task {created_task_count}: prompt {sample_index}")

        # Main loop: wait for task completion and dynamically create new tasks
        try:
            while pending_tasks:
                # Check early stopping status
                if early_stopping_coordinator:
                    should_stop = await early_stopping_coordinator.should_stop_generation.remote()
                    if should_stop:
                        print(
                            f"[AgentLoopWorker] Early stopping signal detected, "
                            f"cancelling {len(pending_tasks)} running tasks and {len(pending_prompts)} pending tasks"
                        )
                        # Cancel all pending tasks
                        for task in pending_tasks:
                            task.cancel()
                        break

                # Wait for any task to complete
                done, still_pending = await asyncio.wait(
                    pending_tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.1,  # Short timeout to periodically check early stopping status
                )

                # Process completed tasks
                for task in done:
                    completed_sample_index = pending_tasks.pop(task)
                    try:
                        sample_index, outputs = await task
                        if not outputs:
                            print(f"[AgentLoopWorker] Prompt {sample_index} completed, but invalid")
                            if early_stopping_coordinator:
                                await early_stopping_coordinator.report_completion.remote(sample_index, is_valid=False)
                        else:
                            completed_outputs[sample_index] = outputs
                            print(f"[AgentLoopWorker] Prompt {sample_index} completed, output {len(outputs)} samples")

                            # Report completion status to coordinator
                            if early_stopping_coordinator:
                                await early_stopping_coordinator.report_completion.remote(sample_index, is_valid=True)

                    except asyncio.CancelledError:
                        print(f"[AgentLoopWorker] Task {completed_sample_index} cancelled")
                    except Exception as e:
                        print(f"[AgentLoopWorker] Task {completed_sample_index} failed: {e}")
                        raise e

                # Create a new task for each completed task (if there are pending prompts)
                new_tasks_created = 0
                for _ in range(len(done)):
                    if pending_prompts and len(pending_tasks) < max_concurrent_tasks:
                        # Check early stopping status to avoid creating new tasks during early stopping
                        if early_stopping_coordinator:
                            should_stop = await early_stopping_coordinator.should_stop_generation.remote()
                            if should_stop:
                                print("[AgentLoopWorker] Early stopping signal detected, stop creating new tasks")
                                break

                        sample_index, group_data = pending_prompts.pop(0)
                        task = asyncio.create_task(
                            self._run_prompt_group(sample_index, group_data, do_filter=not is_validation)
                        )
                        pending_tasks[task] = sample_index
                        new_tasks_created += 1
                        created_task_count += 1
                        print(
                            f"[AgentLoopWorker] Dynamically created new task {created_task_count}: "
                            f"prompt {sample_index}"
                        )

                if new_tasks_created > 0:
                    print(
                        f"[AgentLoopWorker] Created {new_tasks_created} new tasks this round, "
                        f"current running tasks: {len(pending_tasks)}"
                    )

                print(
                    f"[AgentLoopWorker] Task status - "
                    f"Running: {len(pending_tasks)}, "
                    f"Pending: {len(pending_prompts)}, "
                    f"Completed: {len(completed_outputs)}"
                )

        except Exception as e:
            # Ensure all remaining tasks are cancelled on exception
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation to complete
            if pending_tasks:
                await asyncio.gather(*pending_tasks.keys(), return_exceptions=True)
            raise e

        # Only process completed outputs
        if not completed_outputs:
            # If no completed outputs, return empty result
            return None

        all_outputs = []
        for sample_index in sorted(completed_outputs.keys()):
            all_outputs.extend(completed_outputs[sample_index])

        output = self._postprocess(all_outputs)
        return output

    def _group_by_prompt(self, agent_names, raw_prompts, ground_truths, trajectory_info, sampling_params):
        """Group data by prompt sample_index"""
        prompt_groups = defaultdict(list)

        for i, (agent_name, messages, ground_truth, trajectory) in enumerate(
            zip(agent_names, raw_prompts, ground_truths, trajectory_info, strict=True)
        ):
            sample_index = trajectory["sample_index"]
            prompt_groups[sample_index].append(
                {
                    "agent_name": agent_name,
                    "messages": messages,
                    "ground_truth": ground_truth,
                    "trajectory": trajectory,
                    "sampling_params": sampling_params,
                }
            )

        return prompt_groups

    async def _run_prompt_group(self, sample_index: int, group_data: list, do_filter: bool = True):
        """Run all samples for a prompt"""
        tasks = []
        for data in group_data:
            task = asyncio.create_task(
                self._run_agent_loop(
                    data["agent_name"],
                    data["messages"],
                    data["ground_truth"],
                    data["sampling_params"],
                    data["trajectory"],
                )
            )
            tasks.append(task)

        try:
            # Wait for all samples to complete
            outputs = await asyncio.gather(*tasks)

            if any(output is None for output in outputs):
                return sample_index, []

            # Check if all samples have identical reward.reward, if so, this prompt is invalid
            if do_filter and all(output.reward.reward == outputs[0].reward.reward for output in outputs):
                return sample_index, []

            return sample_index, outputs
        except asyncio.CancelledError:
            # Cancel all subtasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for all subtask cancellations to complete
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass  # Ignore exceptions during cancellation

            raise  # Re-raise CancelledError

    async def _run_agent_loop(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        ground_truth: str,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"], sample_index=trajectory["sample_index"], rollout_n=trajectory["rollout_n"]
        ):
            agent_loop = SingleTurnAgentLoop(self.config, self.server_manager, self.tokenizer)
            output = await agent_loop.run(messages, sampling_params)
            if output is None:
                return None
            output.reward = self._compute_reward(ground_truth, output)
            output.rollout_index = trajectory["rollout_index"]
            return output

    def _compute_reward(self, ground_truth: str, output: AgentLoopOutput) -> RewardOutput:
        response_str = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)

        ori_response_str = response_str

        eos_token = self.tokenizer.eos_token
        if response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]
        ret = compute_score(response_str, ground_truth)
        reward = ret["score"]
        acc = ret["acc"]
        pred = ret["pred"]

        # print some samples
        if random.randint(0, 1024) < 1:
            print("\n" + "=" * 80)
            print("🔍 [Debug Sample]")
            print("-" * 80)
            print(f"🤖 Model Response: {ori_response_str}")
            print(f"✅ Ground Truth: {ground_truth}")
            print(f"📊 Evaluation Result: Score={reward:.2f} | Accuracy={acc:.2f} | Prediction={pred}")
            print("=" * 80 + "\n")

        if self.config.actor_rollout_ref.rollout.overlong_buffer.enable:
            overlong_buffer_len = self.config.actor_rollout_ref.rollout.overlong_buffer.len
            expected_len = self.config.actor_rollout_ref.rollout.response_length - overlong_buffer_len
            exceed_len = len(output.response_ids) - expected_len
            overlong_penalty_factor = self.config.actor_rollout_ref.rollout.overlong_buffer.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward

        return RewardOutput(reward=reward, acc=acc, pred=pred)

    def _postprocess(self, inputs: list[AgentLoopOutput]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]
        valid_response_lengths = [len(input.response_ids) for input in inputs]
        print(f"[AgentLoop][DEBUG] max valid_response_lengths: {max(valid_response_lengths)}")

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        for i, input in enumerate(inputs):
            reward_tensor[i, valid_response_lengths[i] - 1] = input.reward.reward
            reward_extra_info["acc"].append(input.reward.acc)
            reward_extra_info["pred"].append(input.reward.pred)
            reward_extra_info["score"].append(input.reward.reward)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
                "token_level_scores": reward_tensor,
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        rollout_index = np.array([input.rollout_index for input in inputs], dtype=np.int32)
        metrics = [input.metrics.model_dump() for input in inputs]
        non_tensor_batch = {
            "__num_turns__": num_turns,
            "rollout_index": rollout_index,
        }
        for key, value in reward_extra_info.items():
            non_tensor_batch[key] = np.array(value)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": metrics})


async def get_trajectory_info(step, index, rollout_index):
    """Get the trajectory info (step, sample_index, rollout_n) asynchrously"""
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append(
            {"step": step, "sample_index": index[i], "rollout_n": rollout_n, "rollout_index": rollout_index[i]}
        )
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_global_server_manager()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_global_server_manager(self):
        """Create global AsyncLLMServerManager as Ray Actor"""
        self.global_load_balancer = GlobalLoadBalancer.options(
            name="global_async_llm_load_balancer",
        ).remote(self.config, self.rollout_dp_size)
        print("[AgentLoopManager] Created global load balancer")

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.global_load_balancer, self.async_llm_servers)
            )
        print(
            f"[AgentLoopManager] Created {len(self.agent_loop_workers)} AgentLoopWorkers, "
            "all using the same global server manager"
        )

    def generate_sequences(self, prompts: DataProto, expected_prompt_num: int = None) -> DataProto:
        """Split input batch and dispatch to agent loop workers with early stopping support.

        Args:
            prompts (DataProto): Input batch.
            expected_prompt_num (int, optional): Expected number of prompts to complete,
                triggers early stopping when reached.

        Returns:
            DataProto: Output batch.
        """
        # print prompts keys for debug
        print(f"[AgentLoopManager] expected_prompt_num: {expected_prompt_num}")
        print(
            f"[AgentLoopManager] prompts keys: {prompts.batch.keys()} "
            f"non_tensor_batch: {prompts.non_tensor_batch.keys()}"
        )

        # Reset global load balancer at the beginning of each generate call
        ray.get(self.global_load_balancer.reset.remote())
        print("[AgentLoopManager] Reset global load balancer")

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        # Create early stopping coordinator (if needed)
        early_stopping_coordinator = None
        if expected_prompt_num is not None:
            early_stopping_coordinator = EarlyStoppingCoordinator.options(
                name=f"early_stopping_coordinator_{uuid4().hex[:8]}"
            ).remote(expected_prompt_num)
            print(f"[AgentLoopManager] Enabled early stopping mechanism: expected_prompt_num={expected_prompt_num}")

        # Group by prompt and assign to workers, ensuring samples from the same prompt go to the same worker
        worker_chunks = self._split_by_prompt(prompts)

        # Start all worker tasks
        worker_tasks = []
        for i, chunk in enumerate(worker_chunks):
            if chunk is not None and len(chunk) > 0:  # Only process non-empty chunks
                task = self.agent_loop_workers[i].generate_sequences.remote(chunk, early_stopping_coordinator)
                worker_tasks.append(task)
            else:
                worker_tasks.append(None)

        # Wait for all workers to complete
        outputs = []
        for i, task in enumerate(worker_tasks):
            if task is not None:
                try:
                    result = ray.get(task)
                    if result is None:
                        continue
                    outputs.append(result)
                except Exception as e:
                    print(f"[AgentLoopManager] Worker {i} execution failed: {e}")
                    # Throw exception early to avoid subsequent computations
                    raise e

        # Terminate all incomplete requests
        self.abort()

        # Merge outputs
        output = DataProto.concat(outputs)
        print(f"[AgentLoopManager] Merged output size: {len(output)}")

        # Get completed prompt set
        completed_prompts = set()
        if early_stopping_coordinator is not None:
            try:
                completed_prompts = ray.get(early_stopping_coordinator.get_completed_prompts.remote())
                print(f"[AgentLoopManager] Early stopping finished, completed {len(completed_prompts)} prompts")
                ray.kill(early_stopping_coordinator)
            except Exception:
                pass

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        if len(outputs) > 0:
            metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
            timing = self._performance_metrics(metrics, output)
        else:
            timing = {}

        output.meta_info = {"timing": timing}
        return output

    def _split_by_prompt(self, prompts: DataProto) -> list:
        """Group by prompt and assign to workers, ensuring samples from the same prompt go to the same worker"""
        # Get sample_index information
        if "index" in prompts.non_tensor_batch:
            indices = prompts.non_tensor_batch["index"]
        else:
            indices = np.arange(len(prompts))

        # Group by sample_index
        prompt_groups = defaultdict(list)
        for i, sample_index in enumerate(indices):
            prompt_groups[sample_index].append(i)

        # Debug info: show prompt distribution
        unique_prompts = list(prompt_groups.keys())
        print(f"[AgentLoopManager] Total {len(unique_prompts)} unique prompts")

        # Assign to workers
        num_workers = len(self.agent_loop_workers)
        worker_assignments = [[] for _ in range(num_workers)]
        worker_prompt_counts = [0] * num_workers  # Track number of prompts assigned to each worker

        # Fix: use sample_index value instead of enumeration order for assignment
        for worker_idx, (sample_index, sample_indices) in enumerate(prompt_groups.items()):
            target_worker = worker_idx % num_workers  # Restore original logic
            worker_assignments[target_worker].extend(sample_indices)
            worker_prompt_counts[target_worker] += 1

        # Create data chunks for each worker
        worker_chunks = []
        for worker_idx in range(num_workers):
            if worker_assignments[worker_idx]:
                indices_to_select = worker_assignments[worker_idx]
                # Create worker's data subset
                chunk = prompts.select_idxs(indices_to_select)
                worker_chunks.append(chunk)
                print(
                    f"[AgentLoopManager] Worker {worker_idx} assigned {len(indices_to_select)} samples "
                    f"({worker_prompt_counts[worker_idx]} prompts)"
                )
            else:
                worker_chunks.append(None)  # This worker has no assigned tasks
                print(f"[AgentLoopManager] Worker {worker_idx} has no assigned tasks")

        return worker_chunks

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def abort(self):
        """Abort all rollout server instances."""
        ray.get([server.abort.remote() for server in self.async_llm_servers])
