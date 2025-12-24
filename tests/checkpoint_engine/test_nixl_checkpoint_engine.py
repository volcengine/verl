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
import pytest
import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine.nixl_checkpoint_engine import NIXLCheckpointEngine


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, role: str) -> None:
        assert role in ["trainer", "rollout"], f"role must be trainer or rollout, but got {role}"
        self.role = role
        self.model = AutoModelForCausalLM.from_pretrained("/mnt/hdfs/wuxibin_wl/model/Qwen3-0.6B")
        self.model.to("cuda")
        bucket_size = 1024 * 1024 * 1024  # 1GB
        self.checkpoint_engine = NIXLCheckpointEngine(bucket_size=bucket_size, device="cuda")
        self.received_weights: dict[str, torch.Tensor] = {}

    def get_metadata(self):
        return self.checkpoint_engine.get_metadata()

    def setup(self, *args, **kwargs):
        self.checkpoint_engine.setup(*args, **kwargs)

    def tear_down(self, *args, **kwargs):
        self.checkpoint_engine.tear_down(*args, **kwargs)

    async def update_weights(self):
        if self.role == "trainer":
            await self.checkpoint_engine.send_weights(self.model.state_dict().items())
            return

        async for name, weight in self.checkpoint_engine.receive_weights():
            self.received_weights[name] = weight.clone()

    def check_weights(self):
        if self.role == "trainer":
            return

        for name, weight in self.model.state_dict().items():
            assert name in self.received_weights, f"weight {name} not received"
            assert torch.allclose(weight, self.received_weights[name]), f"weight {name} not equal"
        self.received_weights.clear()


@pytest.mark.parametrize("num_rollout", [1, 4, 7])
def test_nixl_checkpoint_engine(num_rollout):
    ray.init()
    trainer = Worker.remote("trainer")
    rollout = [Worker.remote("rollout") for _ in range(num_rollout)]
    workers = [trainer] + rollout

    # get metadata of all workers
    metadata = ray.get([worker.get_metadata.remote() for worker in workers])
    kwargs = []
    for rank in range(len(workers)):
        kwargs.append(
            {
                "rank": rank,
                "world_size": len(workers),
                "prev_address": metadata[rank - 1] if rank > 0 else None,
                "next_address": metadata[rank + 1] if rank < len(workers) - 1 else None,
            }
        )

    for _ in range(3):
        # setup communication between all workers
        ray.get([worker.setup.remote(**kwargs[rank]) for rank, worker in enumerate(workers)])

        # update weights of all workers
        ray.get([worker.update_weights.remote() for worker in workers])

        # tear down communication between all workers
        ray.get([worker.tear_down.remote() for worker in workers])

        # check weights of all workers
        ray.get([worker.check_weights.remote() for worker in workers])

    ray.shutdown()
