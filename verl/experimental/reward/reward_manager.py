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
import aiohttp
import json

import ray
from omegaconf import DictConfig

from verl.experimental.reward.reward_loop import get_reward_loop_manager_cls
from verl.protocol import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup

from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RewardLoopWorker:
    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        RewardLoopWork can tackle reward computation:
        (1) rule-based reward computation
        (2) reward model-based reward computation (both disrm and genrm)
        (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

        Reward Computation Logic:
        - if user-customized reward function is provided:
            -> directly use user-customized reward function
        - if user-customized reward function is not provided:
            -> rm is not enabled: use default rule-based reward function
            -> rm is disrm: compute reward score using disrm
            -> rm is genrm: raise error (user-costomized reward func must be provided)

        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        self.reward_fn = get_custom_reward_fn(self.config)
        reward_loop_manager_cls = get_reward_loop_manager_cls(self.config.reward_model.reward_manager)
        self.reward_loop = reward_loop_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
        )

    async def compute_score(self, data: DataProto) -> DataProto:
        if self.config.get("custom_reward_function", None) is not None:
            return await self.reward_loop.run_single(data)
        else:
            if self.config.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data)
            else:
                return await self.reward_loop.run_single(data)

    async def _post_request(self, payload: dict, endpoint: str):
        url = f"http://{self.router_address}/{endpoint}"
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(url, json=payload) as resp:
                output = await resp.text()
                output = json.loads(output)
                return output
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def compute_score_disrm(self, data: DataProto) -> DataProto:
        return await self.reward_loop.run_single(data)


class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    RewardLoopManager will deprecate fsdp/megatron RewardModelWorker in the future.
    """
    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        if self.config.reward_model.enable:
            assert self.config.reward_model.enable_resource_pool is False, "Standalone Reward Model should not initalized with this class."
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self._init_reward_loop_workers()

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward_model.get("num_workers", 1)
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.reward_loop_workers.append(
                RewardLoopWorker.options(
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True,
                    )
                ).remote(self.config, self.reward_router_address)
            )

    # this func is used to replace the legacy fsdp/megatron RewardModelWorker.compute_rm_score
    def compute_rm_score(self, data: DataProto):
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        chunks = data.chunk(len(self.reward_loop_workers))
        outputs = ray.get(
            [
                worker.compute_score.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        output = DataProto.cat(outputs)

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()
        return output

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
