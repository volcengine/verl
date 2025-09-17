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

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, _agent_loop_registry, _DummyConfig
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from verl.experimental.agent_loop.agent_loop import *


class PartialAsyncLLMServerManager(AsyncLLMServerManager):
    async def generate_for_partial(self, request_id, prompt_ids, sampling_params) -> TokenOutput:
        """Generate tokens from prompt ids. with partial rollout function"""
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output


class PartialAgentLoopOutput(AgentLoopOutput):
    """Agent loop output."""

    is_cancel: bool = False
    """Indicates whether the request was interrupted"""
    log_probs: list[float] = None
    """Response token log probs including LLM generated token, tool response token."""


@ray.remote
class PartialAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], rm_executor: BatchExecutor = None
    ):
        self.AsyncLLMServerManager = PartialAsyncLLMServerManager
        super().__init__(config, server_handles, rm_executor)

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> list[AgentLoopOutput]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
            Each AgentLoopOutput contains:
            - prompt_ids: prompt token ids
            - response_ids: response token ids including LLM generated and tool response tokens
            - response_mask: 1 for LLM generated tokens, 0 for tool response tokens
            - num_turns: number of chat turns
            - metrics: performance metrics
        """
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

        # by default, we assume it's a single turn agent
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

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["output"] = partial_output_list[i]
            tasks.append(
                asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
            )
        return await asyncio.gather(*tasks)

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
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
            return await agent_loop.run(sampling_params, **kwargs)


class PartialAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        self.AgentLoopWorker = PartialAgentLoopWorker
        super().__init__(config, worker_group, rm_wg)

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> list[AgentLoopOutput]:
        """
        异步处理单个样本, 需要复制n次

        Args:
            sample: 单个样本数据
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            tuple[AgentLoopOutput, float]: 处理结果和处理时间
        """
        # 使用负载均衡选择 worker
        worker = self._select_best_worker()
        # 异步处理单个样本 - 使用无后处理版本获取原始AgentLoopOutput
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """选择最佳的 worker（简单的轮询负载均衡）"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    async def sleep(self):
        futures = [replica.sleep.remote() for replica in self.rollout_replicas]
        await asyncio.gather(*[asyncio.wrap_future(future.future()) for future in futures], return_exceptions=True)

    async def wake_up(self):
        futures = [replica.wake_up.remote() for replica in self.rollout_replicas]
        await asyncio.gather(*[asyncio.wrap_future(future.future()) for future in futures], return_exceptions=True)

    async def cancel_async(self):
        """Cancel all rollout tasks asynchronously."""
        futures = [replica.cancel.remote() for replica in self.rollout_replicas]
        await asyncio.gather(*[asyncio.wrap_future(future.future()) for future in futures], return_exceptions=True)

    async def resume_async(self):
        """Cancel all rollout tasks asynchronously."""
        futures = [replica.resume.remote() for replica in self.rollout_replicas]
        await asyncio.gather(*[asyncio.wrap_future(future.future()) for future in futures], return_exceptions=True)

    def _run_all(self, tasks: list[asyncio.Task]):
        pass
