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

import ray

from verl import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.config import RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class
from verl.workers.rollout.utils import get_free_port

from .sglang_router import SGLangRouter

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    def __init__(self, config: RewardModelConfig, worker_group: RayWorkerGroup = None):
        """
        Initialize the reward model manager.

        Args:
            config (RewardModelConfig): Reward model configuration.
            worker_group (RayWorkerGroup, optional): Worker group. Defaults to None.
        """
        self.config = config
        self.worker_group = worker_group
        self._initialize_llm_servers()
        self._initialize_router()
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        assert self.config.rollout.name == "sglang", "Only sglang is supported now"
        rollout_world_size = self.config.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(self.config.rollout.name)
        rollout_config = self.config.rollout
        model_config = self.config.model
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_colocated(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_router(self):
        router_ip = ray.util.get_node_ip_address()
        router_port, _ = get_free_port(router_ip)
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        # current implementation only support sglang
        assert self.config.rollout.name == "sglang", "Only sglang is supported now"

        router = SGLangRouter.options(
            name="reward_model_router",
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(
            router_ip=router_ip,
            router_port=router_port,
            worker_urls=worker_urls,
            balance_abs_threshold=4,
        )
        self.router_handle = router

    def get_handle(self):
        return self.router_handle

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())

    # just for test purpose
    def generate_sequences(self, prompts: DataProto, sampling_params: dict):
        router_inputs = [
            {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in prompts.non_tensor_batch.get("raw_prompt_ids")
        ]
        responses = ray.get(
            [
                self.router_handle.generate.remote(
                    prompt_ids=router_input["prompt_token_ids"],
                    sampling_params=sampling_params,
                )
                for router_input in router_inputs
            ]
        )

        return responses
