# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import signal
import socket
import threading
import time
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.fs import copy_to_local


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    创建资源池管理器

    Args:
        config: 配置对象
        roles: 需要创建资源池的角色列表

    Returns:
        ResourcePoolManager: 资源池管理器
    """
    # 构建资源池规格
    resource_pool_spec = {}
    mapping = {}

    # Actor/Critic资源池（训练相关）
    if any(role in roles for role in [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        # 训练相关角色映射到同一个资源池
        for role in [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Rollout资源池
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
        resource_pool_spec["rollout_pool"] = rollout_pool
        mapping[Role.Rollout] = "rollout_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    """
    创建角色到worker类的映射

    Args:
        config: 配置对象

    Returns:
        dict: 角色到worker类的映射
    """
    # 根据策略选择worker类
    if config.actor_rollout_ref.actor.strategy == "fsdp2":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from recipe.one_step_off_policy.fsdp_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
            RolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from recipe.one_step_off_policy.megatron_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
            RolloutWorker,
        )
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        )
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

    role_worker_mapping = {
        Role.Actor: ray.remote(actor_rollout_cls),
        Role.Rollout: ray.remote(RolloutWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    # 添加reward model（如果启用）
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp2":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")

        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    # 添加reference policy（如果需要KL loss或reward）
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    return role_worker_mapping, ray_worker_group_cls


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        """运行完全异步的PPO训练"""
        print("Starting fully async PPO training...")
        # 初始化基础组件
        self._initialize_components(config)
        # 启动训练流程
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        """
        初始化所有组件

        Args:
            config: 配置对象

        Returns:
            bool: 是否初始化成功
        """
        # 打印配置信息
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # 初始化模型路径和tokenizer
        print("Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config  # 保存config以供其他方法使用

        # 创建worker映射和资源池
        print("Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # 创建奖励函数
        print("Loading reward functions...")
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        self.components["reward_fn"] = reward_fn
        self.components["val_reward_fn"] = val_reward_fn

        # 创建MessageQueue
        self.max_queue_size = (
                config.async_training.staleness_threshold
                * config.data.train_batch_size
                * config.actor_rollout_ref.rollout.n
        ) * 10 # x 10 避免死锁
        print("Creating MessageQueue...")
        message_queue = MessageQueue.remote(config, self.max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)

        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        # 创建Rollouter
        print("Creating FullyAsyncRollouter...")
        self._create_rollouter(config)

        # 创建Trainer
        print("Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        # 设置参数同步
        print("Setting up parameter synchronization...")
        from recipe.fully_async_policy.param_sync import ParameterSynchronizer

        param_synchronizer = ParameterSynchronizer.remote(
            config=config,
            trainer=self.components["trainer"],
            rollouter=self.components["rollouter"],
            mq=self.components["message_queue_client"],
        )

        # 将参数同步器设置到trainer和rollouter
        ray.get(self.components["trainer"].set_parameter_synchronizer.remote(param_synchronizer))
        ray.get(self.components["rollouter"].set_parameter_synchronizer.remote(param_synchronizer))

        # 首先同步一次参数
        ray.get(param_synchronizer.sync_weights.remote(0))

        self.components["param_synchronizer"] = param_synchronizer
        print("All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        """创建Rollouter"""
        pprint(self.components)
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={Role.Rollout: self.components["role_worker_mapping"][Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            max_queue_size=self.max_queue_size,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_message_queue_client.remote(self.components["message_queue_client"]))
        self.components["rollouter"] = rollouter
        print("Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        """创建Trainer"""
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            reward_fn=self.components["reward_fn"],
            val_reward_fn=self.components["val_reward_fn"],
            device_name=config.trainer.device,
        )

        # 初始化Trainer
        ray.get(trainer.init_workers.remote())
        ray.get(trainer.set_message_queue_client.remote(self.components["message_queue_client"]))
        self.components["trainer"] = trainer
        print("FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self):
        """运行训练循环"""
        self.running = True

        print("Starting Rollouter in background...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        ray.get(rollouter_future)
        ray.get(trainer_future)

        self.components["message_queue_client"].clear_queue()

        print("Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    """主入口函数"""
    from verl.trainer.main_ppo import run_ppo

    # 确保异步训练配置存在
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)


if __name__ == "__main__":
    main()
