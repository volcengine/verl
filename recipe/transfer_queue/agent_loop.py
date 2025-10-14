# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import numpy as np
import ray
from transfer_queue import BatchMeta

import verl.experimental.agent_loop.agent_loop as agent_loop
from verl import DataProto


class AgentLoopManager(agent_loop.AgentLoopManager):
    def generate_sequences(self, prompts: BatchMeta) -> BatchMeta:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (BatchMeta): Input batch.

        Returns:
            BatchMeta: Output batch metadata.
        """

        if self.rm_micro_batch_size and len(prompts) % self.rm_micro_batch_size != 0:
            raise ValueError(
                f"The length of prompts {len(prompts)} cannot divide the world size of rm_wg {self.rm_micro_batch_size}"
            )
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = BatchMeta.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.extra_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.set_extra_info("timing", timing)
        return output

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

        return timing

    def create_transferqueue_client(self, controller_infos, storage_infos, role):
        ray.get(
            [
                worker.create_transferqueue_client.remote(controller_infos, storage_infos, role)
                for worker in self.agent_loop_workers
            ]
        )
