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
"""
e2e test verl.single_controller.ray
"""
import os
import ray

from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.decorator import register, Dispatch, collect_all_to_all, Execute


@ray.remote
class TestActor(Worker):

    def __init__(self) -> None:
        super().__init__()

    def getenv(self, key):
        val = os.getenv(key, f"{key} not set")
        return val


def test_basics():
    ray.init()

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    worker_group = RayWorkerGroup(resource_pool=resource_pool,
                                  ray_cls_with_init=class_with_args,
                                  name_prefix="worker_group_basic")

    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
    assert output == ["4", "4", "4", "4"]

    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_RANK")
    assert set(output) == set(["0", "1", "2", "3"])

    ray.shutdown()


if __name__ == '__main__':
    test_basics()
