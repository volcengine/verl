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
from omegaconf import DictConfig

from verl.experimental.agent_loop import AgentLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.config import RewardModelConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

if os.environ["LEGACY_IMPL_RM"] == "disable":
    from verl.workers.roles import RewardModelWorker
else:
    from verl.workers.fsdp_workers import RewardModelWorker


def init_agent_loop_manager(
    config: DictConfig, reward_model_config: RewardModelConfig = None
) -> AgentLoopManager | RayWorkerGroup:
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    actor_rollout_cls = (
        AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    )
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
    }
    reward_model_config = reward_model_config or config.reward_model
    if reward_model_config.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    if reward_model_config.enable_resource_pool:
        mapping[Role.RewardModel] = "reward_pool"
        if reward_model_config.n_gpus_per_node <= 0:
            raise ValueError("reward_model_config.n_gpus_per_node must be greater than 0")
        if reward_model_config.nnodes <= 0:
            raise ValueError("reward_model_config.nnodes must be greater than 0")

        reward_pool = [reward_model_config.n_gpus_per_node] * reward_model_config.nnodes
        resource_pool_spec["reward_pool"] = reward_pool
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout"
    )
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    if reward_model_config.enable:
        # we create a RM here
        resource_pool = resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(role_worker_mapping[Role.RewardModel], config=reward_model_config)
        resource_pool_to_cls[resource_pool]["rm"] = rm_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    if config.actor_rollout_ref.rollout.mode == "sync":
        return actor_rollout_wg

    if reward_model_config.enable_resource_pool and reward_model_config.enable:
        rm_wg = all_wg["rm"]
        rm_wg.init_model()
    else:
        rm_wg = None
    # =========================== 2. Create AgentLoopManager ===========================
    agent_loop_manager = AgentLoopManager(
        config=config,
        worker_group=actor_rollout_wg,
        rm_wg=rm_wg,
    )

    return agent_loop_manager


"""
LEGACY_IMPL_RM=enable python tests/experimental/agent_loop/test_agent_loop_reward_model.py
tensor([ 0.8672,  0.6016,  0.8086,  0.2051,  0.9141,  1.0234, -0.4727,  0.5703,
         0.4395,  0.3691,  0.9219,  0.5664,  0.9414,  0.7812,  0.7852,  1.0156,
        -0.6953,  1.0156,  0.1504,  0.0289,  1.1797, -0.5156, -0.3906,  0.8477,
        -0.1240,  0.9102, -0.4551,  0.4434,  1.3047,  0.3711,  1.0469, -0.2832,
         0.9336,  1.0703,  0.9648,  0.7383,  0.9062, -0.3438,  0.6758,  0.5625,
         0.5664,  1.1953,  0.9766,  1.1328,  1.2734,  0.0435,  1.4297,  1.1797,
         1.3125,  1.0156, -0.0801,  1.2344,  0.8984,  0.8750,  0.9727,  0.6641,
         0.7930,  1.0547,  0.6641,  0.6758,  1.1016,  0.6914,  0.3477,  0.3906,
         0.9023,  1.2266,  1.0625,  0.2617,  0.0728,  0.7578,  1.0859,  1.1562,
         1.0391, -0.0889, -0.8125,  0.8359,  1.3359, -0.5234,  0.6914,  1.1172,
         0.9375,  1.1484,  0.9688,  1.3125, -0.4160,  0.2656,  0.9023,  1.3672,
         0.7969,  1.4297,  1.4531,  0.9375,  0.9336,  1.0312,  1.1250,  0.9336,
         1.3047,  0.9688,  0.8125,  1.0469, -0.1924,  0.9609,  1.2422,  0.8672,
         1.2500, -0.8203,  0.6328, -0.8359, -0.0830,  0.0618,  1.3359,  1.0156,
         0.6602, -0.3887,  0.8203,  0.9648,  1.2656,  1.0781,  0.9336,  1.0469,
         1.0469,  1.0391,  0.9219,  0.0131,  0.7305,  1.2031,  1.5859,  0.7930])
LEGACY_IMPL_RM=disable python tests/experimental/agent_loop/test_agent_loop_reward_model.py

"""
