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

import ray

from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# Mixin that adds a simple id() method
class IDMixin:
    def id(self):
        return 1


# Additional mixin for RayWorkerGroup
class ExtraMixin:
    def extra(self):
        return 2


@ray.remote
class TestActor(Worker):
    # TestActor Class with is a Ray Actor
    def __init__(self, x) -> None:
        super().__init__()
        self._x = x


def test_workergroup_with_single_mixin():
    # Mix IDMixin into the base WorkerGroup
    wg = WorkerGroup.with_mixin(IDMixin)(resource_pool=None)

    # It should still behave like a WorkerGroup
    assert isinstance(wg, WorkerGroup)
    # And pick up the mixin
    assert hasattr(wg, "id")
    assert wg.id() == 1


def test_rayworkergroup_with_two_mixins():
    ray.init(num_cpus=100)
    resource_pool = RayResourcePool([4], use_gpu=False)
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
    # Mix both IDMixin and ExtraMixin into RayWorkerGroup
    rwg = RayWorkerGroup.with_mixin(IDMixin, ExtraMixin)(resource_pool=resource_pool, ray_cls_with_init=class_with_args)

    # It should still behave like a RayWorkerGroup
    assert isinstance(rwg, RayWorkerGroup)
    # Mixin methods should be available
    assert hasattr(rwg, "id")
    assert rwg.id() == 1

    assert hasattr(rwg, "extra")
    assert rwg.extra() == 2
