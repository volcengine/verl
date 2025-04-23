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

import os
import torch
import torch.distributed.rpc as rpc

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.torchrpc import (
    TorchRPCResourcePool,
    TorchRPCClassWithInitArgs,
    TorchRPCWorkerGroup,
    create_colocated_worker_cls,
    rref_to_here
)


class Actor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch["a"] += self.rank
        return data


class Critic(Worker):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def sub(self, data: DataProto):
        data.batch["a"] -= self.config["b"]
        return data


def test_colocated_workers():
    # MASTER_ADDR, MASTER_PORT, TORCHRPC_RANK, TORCHRPC_WORLD_SIZE must be set first
    # Then run this code on every node
    rank = int(os.environ.get('TORCHRPC_RANK'))
    world_size = int(os.environ.get('TORCHRPC_WORLD_SIZE'))
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    if rank == 0:
        data = DataProto.from_dict({"a": torch.zeros(10)})
        # create separate workers on the same resource pool
        actor_cls = TorchRPCClassWithInitArgs(cls=Actor)
        critic_cls = TorchRPCClassWithInitArgs(cls=Critic, config={"b": 10})
        resource_pool = TorchRPCResourcePool(process_on_nodes=[2])

        actor_wg = TorchRPCWorkerGroup(resource_pool=resource_pool, cls_with_init=actor_cls)
        critic_wg = TorchRPCWorkerGroup(resource_pool=resource_pool, cls_with_init=critic_cls)

        expected_actor_output = actor_wg.add(data)
        expected_critic_output = critic_wg.sub(data)

        # create colocated workers
        cls_dict = {"actor": actor_cls, "critic": critic_cls}
        cls_with_init = create_colocated_worker_cls(cls_dict)
        wg_dict = TorchRPCWorkerGroup(resource_pool=resource_pool, cls_with_init=cls_with_init)



        actor_output = wg_dict.actor_add(data)
        critic_output = wg_dict.critic_sub(data)
        torch.testing.assert_close(expected_actor_output.batch, actor_output.batch, atol=0, rtol=0)
        torch.testing.assert_close(expected_critic_output.batch, critic_output.batch, atol=0, rtol=0)
    rpc.shutdown()

if __name__ == "__main__":
    test_colocated_workers()
