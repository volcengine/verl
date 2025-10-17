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
import json
import logging
import os

import aiohttp

from verl import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    """Reward model manager."""

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
        model_config = HFModelConfig(
            path=self.config.model.path,
            external_lib=self.config.model.external_lib,
            trust_remote_code=self.config.model.trust_remote_code,
        )
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
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        if self.config.rollout.name == "sglang":
            from .router.sglang_router import launch_router_process
        else:
            from .router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self):
        return self.router_address

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())

    # just for test purpose
    async def generate(self, prompt_token_ids: list[int], sampling_params: dict):
        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": sampling_params,
        }
        url = f"http://{self.router_address}/generate"
        try:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
            async with session.post(url, json=payload) as resp:
                output = await resp.text()
                output = json.loads(output)
                return output
        except Exception as e:
            logger.error(f"Error in generate_single: {e}")
            raise e
        finally:
            await session.close()

    def generate_sequences(self, prompts: DataProto, sampling_params: dict):
        router_inputs = [
            {"prompt_token_ids": raw_prompt_ids, "sampling_params": sampling_params}
            for raw_prompt_ids in prompts.non_tensor_batch.get("raw_prompt_ids")
        ]
        tasks = [self.generate(**router_input) for router_input in router_inputs]
        responses = self._run_all(tasks)
        return responses
