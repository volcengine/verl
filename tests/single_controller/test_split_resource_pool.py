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

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)


@ray.remote
class Actor(Worker):
    def __init__(self, worker_id) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.temp_tensor = torch.rand(1024, 1024).to("cuda")

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

        assert torch.distributed.get_world_size() == self.world_size
        assert torch.distributed.get_rank() == self.rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch["a"] += self.rank + self.worker_id
        return data


def test_split_resource_pool():
    ray.init()
    # assume we have 2 nodes, with 4 GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[4, 4])
    global_resource_pool.get_placement_groups()

    # first 4 gpus for actor_1, last 4 gpus for actor_2
    actor_resource_pool, critic_resource_pool = global_resource_pool.split(split_size=4)
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_resource_pool,
        ray_cls_with_init=actor_cls_1,
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_resource_pool,
        ray_cls_with_init=actor_cls_2,
    )
    assert actor_worker_1.world_size == 4
    assert actor_worker_2.world_size == 4

    data = DataProto.from_dict({"a": torch.zeros(8)})
    actor_output_1 = actor_worker_1.add(data)
    actor_output_2 = actor_worker_2.add(data)
    assert actor_output_1.batch["a"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert actor_output_2.batch["a"].tolist() == [100, 100, 101, 101, 102, 102, 103, 103]

    ray.shutdown()
