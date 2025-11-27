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

import ray

from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServerBase,
    vLLMReplica,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class OneStepOffVLLMHttpServer(vLLMHttpServerBase):
    async def reset_prefix_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()


class OneStepOffLLMReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = OneStepOffVLLMHttpServer

    async def reset_prefix_cache(self):
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.reset_prefix_cache.remote() for server in self.servers])
