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
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig

from recipe.fully_async_policy.vllm_rollout.vllm_async_server import FullyAsyncvLLMReplica
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    BatchExecutor,
    _agent_loop_registry,
    _DummyConfig,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.rollout_trace import rollout_trace_attr
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    async def generate_for_partial(self, request_id, prompt_ids, sampling_params) -> TokenOutput:
        """Generate tokens from prompt ids. with partial rollout function"""
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output


class FullyAsyncAgentLoopOutput(AgentLoopOutput):
    """Agent loop output."""

    is_cancel: bool = False
    """Indicates whether the request was interrupted"""
    log_probs: list[float] = None
    """Response token log probs including LLM generated token, tool response token."""
    param_version_start: int = 0
    """Indicate start parameter version when this response is generated"""
    param_version_end: int = 0
    """Indicate end parameter version when this response is generated, used for partial rollout"""


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], rm_executor: BatchExecutor = None
    ):
        self.server_manager_class = FullyAsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, rm_executor)

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> list[AgentLoopOutput]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[FullyAsyncAgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
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
                server_manager=self.server_manager_class,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            return await agent_loop.run(sampling_params, **kwargs)


class PartialAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        # 初始化基本属性，但不执行异步操作
        self.config = config
        self.worker_group = worker_group
        self.rm_executor = None
        self.rm_micro_batch_size = None
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        self.rollout_replica_class = FullyAsyncvLLMReplica

        # 初始化其他必要属性为None，稍后在异步初始化中设置
        self.rm_wg = rm_wg
        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_workers = None

    @classmethod
    async def create(cls, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        """异步工厂方法来创建和初始化 PartialAgentLoopManager 实例"""
        instance = cls(config, worker_group, rm_wg)
        await instance._async_init()
        return instance

    async def _async_init(self):
        """异步初始化方法"""
        # 处理 rm_wg 相关初始化
        if self.rm_wg:

            def batch_fn(data_list: list[DataProto]) -> list[torch.Tensor]:
                new_data_list = []
                for data in data_list:
                    temp_non_tensor_batch = {"__num_turns__": data.non_tensor_batch["__num_turns__"]}
                    temp_data = DataProto(batch=data.batch, non_tensor_batch=temp_non_tensor_batch)
                    new_data_list.append(temp_data)

                new_batch = DataProto.concat(new_data_list)
                out_data = self.rm_wg.compute_rm_score(new_batch)
                return out_data.split(1)

            self.rm_executor = BatchExecutor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(batch_fn, self.rm_wg.world_size)

            self.rm_micro_batch_size = self.rm_wg.world_size

        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()

    async def _initialize_llm_servers_async(self):
        """异步初始化 LLM 服务器"""
        rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank, config=self.config, gpus_per_node=self.config.trainer.n_gpus_per_node
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> list[AgentLoopOutput]:
        """
        异步处理单个样本

        Args:
            sample: 单个样本数据
            partial_output_list: Optional[List[AgentLoopOutput]]: 已经 rollout 的结果

        Returns:
            list[AgentLoopOutput]: 处理结果列表
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """选择最佳的 worker（简单的轮询负载均衡）"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])
