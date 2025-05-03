'''
Author: [Mr.L]
Date: 2025-05-03 15:48:12
LastEditors: [Mr.l]
LastEditTime: 2025-05-03 15:48:54
Description: 
'''
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

import ray
import torch

from verl.single_controller.base.decorator import Dispatch, Execute, collect_all_to_all, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    """
    for arg in args:
        assert len(arg) == 2
        for i in range(worker_group.world_size - 2):
            arg.append(arg[i % 2])
    for k, v in kwargs.items():
        assert len(v) == 2
        for i in range(worker_group.world_size - 2):
            v.append(v[i % 2])
    return args, kwargs


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, x) -> None:
        super().__init__()
        self._x = x

    def foo(self, y):
        return self._x + y

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def foo_rank_zero(self, x, y):
        return self._x + y + x

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def foo_one_to_all(self, x, y):
        return self._x + y + x

    @register(Dispatch.ALL_TO_ALL, blocking=False)
    def foo_all_to_all(self, x, y):
        return self._x + y + x

    @register(dispatch_mode={"dispatch_fn": two_to_all_dispatch_fn, "collect_fn": collect_all_to_all})
    def foo_custom(self, x, y):
        return self._x + y + x


def test_basics():
    ray.init()

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=False)
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)

    worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic")

    print(worker_group.worker_names)

    # this will wait for all the results
    output = worker_group.execute_all_sync("foo", y=3)
    assert output == [5, 5, 5, 5]

    # this is a list of object reference. It won't block.
    output_ref = worker_group.execute_all_async("foo", y=4)
    print(output_ref)

    assert ray.get(output_ref) == [6, 6, 6, 6]

    output_ref = worker_group.foo_one_to_all(x=1, y=2)
    assert ray.get(output_ref) == [5, 5, 5, 5]

    output_ref = worker_group.foo_all_to_all(x=[1, 2, 3, 4], y=[5, 6, 7, 8])
    assert ray.get(output_ref) == [8, 10, 12, 14]

    ray.shutdown()


if __name__ == "__main__":
    test_basics()
