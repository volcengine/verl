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

import time

import ray

from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup, merge_resource_pool
from verl.single_controller.base.worker import Worker


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, cuda_visible_devices=None) -> None:
        super().__init__(cuda_visible_devices)

    def get_node_id(self):
        return ray.get_runtime_context().get_node_id()


def test():
    ray.init()

    # test single-node-no-partition
    print(f"test single-node-no-partition")
    resource_pool = RayResourcePool([8], use_gpu=True)

    class_with_args = RayClassWithInitArgs(cls=TestActor)

    print("create actor worker group")
    actor_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="high_level_api_actor")
    print("create critic worker group")
    critic_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="hight_level_api_critic")
    print("create rm worker group")
    rm_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="high_level_api_rm")
    print("create ref worker group")
    ref_wg = RayWorkerGroup(resource_pool, class_with_args, name_prefix="high_level_api_ref")

    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]

    del actor_wg
    del critic_wg
    del rm_wg
    del ref_wg

    [ray.util.remove_placement_group(pg) for pg in resource_pool.get_placement_groups()]
    print("wait 5s to remove placemeng_group")
    time.sleep(5)
    # test single-node-multi-partition

    print(f"test single-node-multi-partition")
    rm_resource_pool = RayResourcePool([4], use_gpu=True, name_prefix="rm")
    ref_resource_pool = RayResourcePool([4], use_gpu=True, name_prefix="ref")
    total_resource_pool = merge_resource_pool(rm_resource_pool, ref_resource_pool)

    assert rm_resource_pool.world_size == 4
    assert ref_resource_pool.world_size == 4
    assert total_resource_pool.world_size == 8

    actor_wg = RayWorkerGroup(total_resource_pool, class_with_args, name_prefix="high_level_api_actor")
    critic_wg = RayWorkerGroup(total_resource_pool, class_with_args, name_prefix="high_level_api_critic")
    rm_wg = RayWorkerGroup(rm_resource_pool, class_with_args, name_prefix="high_level_api_rm")
    ref_wg = RayWorkerGroup(ref_resource_pool, class_with_args, name_prefix="high_level_api_ref")

    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4)]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4, 8)]

    ray.shutdown()
