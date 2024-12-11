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

from verl.single_controller.remote import remote, RemoteBackend, SharedResourcePool
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.base.worker import Worker


@remote(process_on_nodes=[3], use_gpu=True, name_prefix="actor", sharing=SharedResourcePool)
class Actor(Worker):
    ...


@remote(process_on_nodes=[3], use_gpu=True, name_prefix="critic", sharing=SharedResourcePool)
class Critic(Worker):
    ...


@remote(process_on_nodes=[2], use_gpu=True, name_prefix="reward", sharing=SharedResourcePool.from_role("actor"))
class Reward(Worker):
    ...


@remote(process_on_nodes=[2], use_gpu=True, name_prefix="ref", sharing=SharedResourcePool.from_role("actor", "critic"))
class Ref(Worker):
    ...


@remote(process_on_nodes=[1], use_gpu=True, name_prefix="sec_rm", sharing=SharedResourcePool.from_role("any"))
class SecRM(Worker):
    ...


def test():
    print("Remote.init_distributed")
    remote.init_distributed(backend=RemoteBackend.RAY)

    print("create actor worker group")
    actor = Actor()

    print("create critic worker group")
    critic = Critic()

    print("create rm worker group")
    reward = Reward()

    print("create ref worker group")
    ref = Ref()

    print("create sec_rm worker group")
    sec_rm = SecRM()

    actor_gpus = actor.execute_all_sync("get_cuda_visible_devices")
    critic_gpus = critic.execute_all_sync("get_cuda_visible_devices")
    reward_gpus = reward.execute_all_sync("get_cuda_visible_devices")
    ref_gpus = ref.execute_all_sync("get_cuda_visible_devices")
    sec_rm_gpus = sec_rm.execute_all_sync("get_cuda_visible_devices")

    for gpu in actor_gpus:
        assert gpu not in critic_gpus, f"actor gpus = {actor_gpus}, critic gpus = {critic_gpus}"

    for gpu in critic_gpus:
        assert gpu not in actor_gpus, f"actor gpus = {actor_gpus}, critic gpus = {critic_gpus}"

    for gpu in reward_gpus:
        assert gpu in actor_gpus, f"actor gpus = {actor_gpus}, reward gpus = {reward_gpus}"

    for gpu in ref_gpus:
        assert gpu in actor_gpus + critic_gpus, \
            f"actor gpus = {actor_gpus}, critic gpus = {critic_gpus}, ref gpus = {ref_gpus}"

    for gpu in sec_rm_gpus:
        assert gpu in actor_gpus + critic_gpus, \
            f"actor gpus = {actor_gpus}, critic gpus = {critic_gpus}, sec rm gpus = {sec_rm_gpus}"

    # for ci only
    import ray
    ray.shutdown()
