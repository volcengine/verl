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

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

import ray
import torch
import torch.distributed
import torch.distributed.rpc as rpc

from verl.single_controller.base.worker import Worker
from verl.single_controller.torchrpc import TorchRPCClassWithInitArgs, TorchRPCResourcePool, TorchRPCWorkerGroup, rref_to_here, torchrpc_remote

class TestAllGatherActor(Worker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def init(self):
        os.environ["NCCL_DEBUG"]="WARN"
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
        print(
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_WORLD_SIZE"],
            os.environ["LOCAL_RANK"],
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["CUDA_VISIBLE_DEVICES"],
        )
        torch.distributed.init_process_group()
        self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device="cuda")
        self.tensor += self.rank

    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,), dtype=self.tensor.dtype, device=self.tensor.device
        )
        torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output


@ray.remote
class TestAllGatherActorV2(Worker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

        torch.distributed.init_process_group()
        self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device="cuda")
        self.tensor += self.rank

    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,), dtype=self.tensor.dtype, device=self.tensor.device
        )
        torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output

@torchrpc_remote
def test_all_gather_torch():
    """
    In this test, we instantiate 4 GPUs in a group and test the all_gather
    """
    resource_pool = TorchRPCResourcePool([1, 1], use_gpu=True)
    class_with_args = TorchRPCClassWithInitArgs(cls=TestAllGatherActor, size=2)

    worker_group = TorchRPCWorkerGroup(resource_pool, class_with_args, name_prefix="worker_group_torch")

    worker_group.execute_all_sync("init")
    output = worker_group.execute_all_sync("all_gather")
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0].cpu()
    print(output)
    assert torch.all(output == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64))
    rpc.shutdown()


def test_all_gather_torch_v2():
    """
    In this test, we instantiate 4 GPUs in a group and test the all_gather
    """
    ray.init()

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestAllGatherActorV2, size=2)

    worker_group = RayWorkerGroup(resource_pool, class_with_args, name_prefix="worker_group_torch")

    output = worker_group.execute_all_sync("all_gather")
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0].cpu()
    print(output)
    assert torch.all(output == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64))

    ray.shutdown()


if __name__ == "__main__":
    test_all_gather_torch()