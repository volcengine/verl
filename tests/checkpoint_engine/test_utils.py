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
import os

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.utils.fs import copy_to_local


@ray.remote(num_gpus=1)
class Worker:
    def __init__(
        self,
        role: str,
        checkpoint_backend: str,
        checkpoint_kwargs: dict,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: str = "12345",
    ) -> None:
        assert role in ["trainer", "rollout"], f"role must be trainer or rollout, but got {role}"
        self.role = role
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

        model_path = os.environ["HDFS_ROOT"] + "/model/Qwen3-8B-Base"
        local_path = copy_to_local(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16)
        self.model.to("cuda")
        self.checkpoint_engine = CheckpointEngineRegistry.new(checkpoint_backend, **checkpoint_kwargs)
        self.received_weights: dict[str, torch.Tensor] = {}

    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

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
