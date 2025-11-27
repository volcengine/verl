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

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
)
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from recipe.one_step_off_policy.vllm_rollout.vllm_async_server import OneStepOffLLMReplica


class OneStepOffAgentLoopManager(AgentLoopManager):
    def __init__(
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    ):
        self.rollout_replica_class = OneStepOffLLMReplica
        super().__init__(config, worker_group, rm_resource_pool)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # remove wake_up()/sleep() methods internally check

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # Use asyncio.gather with ray.get wrapped in asyncio.to_thread to avoid blocking
        import asyncio

        outputs = await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, worker.generate_sequences.remote(chunk))
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def reset_prefix_cache(self):
        await asyncio.gather(*[replica.reset_prefix_cache() for replica in self.rollout_replicas])
