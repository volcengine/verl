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

import logging
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
from recipe.fully_async_policy.param_sync import AsyncParameterSynchronizer
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
        # 设置信号处理
        self._setup_signal_handlers()
        # 初始化基础组件
        self._initialize_components(config)
        time.sleep(60)
        # 启动训练流程
        # self._run_training_loop()

        # self._cleanup_resources()

    def _setup_signal_handlers(self):
        """设置信号处理器"""

        def signal_handler(signum, frame):
            print(f"Received signal {signum}, initiating shutdown...")
            self.running = False
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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
        print("Creating MessageQueue...")
        max_queue_size = config.async_training.get("max_queue_size", 1000)
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)

        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        # 创建Rollouter
        print("Creating Rollouter...")
        self._create_rollouter(config)

        # 创建Trainer
        print("Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        # 设置参数同步
        # print("Setting up parameter synchronization...")
        # param_synchronizer = AsyncParameterSynchronizer(
        #     config=config,
        #     actor_wg=self.components["trainer"].actor_wg,
        #     rollouter=self.components["rollouter"],
        # )
        # self.components["param_synchronizer"] = param_synchronizer
        # print("All components initialized successfully")

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
        )
        print(rollouter)

        print("========== rollouter init workers ======")

        # 初始化Rollouter
        ray.get(rollouter.init_workers.remote())

        ray.get(rollouter.set_message_queue_client.remote(self.components["message_queue_client"]))

        self.components["rollouter"] = rollouter
        print("Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        """创建Trainer"""
        # 创建trainer角色映射（排除Rollout）
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
        self._monitor_components()
        ray.get(rollouter_future)
        ray.get(trainer_future)

        print("Training completed or interrupted")

    def _run_rollouter(self):
        try:
            ray.get(self.components["rollouter"].fit.remote())
        except Exception as e:
            print(f"Rollouter error: {e}")
            self.running = False
            self.shutdown_event.set()

    def _run_trainer(self):
        """运行trainer"""
        try:
            self.components["trainer"].fit()
        except Exception as e:
            print(f"Trainer error: {e}")
        finally:
            self.running = False
            self.shutdown_event.set()

    def _monitor_components(self):
        """监控组件状态"""
        print("Starting component monitoring...")

        last_stats_time = time.time()
        stats_interval = 60.0  # 60秒报告一次统计

        while self.running and not self.shutdown_event.is_set():
            try:
                # 等待一段时间或直到收到停止信号
                if self.shutdown_event.wait(timeout=10.0):
                    break

                # 定期报告统计信息
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    self._log_component_statistics()
                    last_stats_time = current_time

                # 检查组件健康状态
                self._check_component_health()

            except Exception as e:
                print(f"Error in component monitoring: {e}")

        print("Component monitoring stopped")

    def _log_component_statistics(self):
        """记录组件统计信息"""
        try:
            # 获取Trainer统计
            trainer_stats = self.components["trainer"].get_statistics()

            # 获取Rollouter统计
            rollouter_stats = ray.get(self.components["rollouter"].get_statistics.remote(), timeout=5.0)

            # 获取队列统计
            queue_stats = self.components["message_queue_client"].get_statistics()

            print("=== Component Statistics ===")
            print(
                f"Trainer - Steps: {trainer_stats['global_steps']}, "
                f"Samples: {trainer_stats['processed_samples']}, "
                f"Param version: {trainer_stats['current_param_version']}"
            )

            print(
                f"Rollouter - Generated: {rollouter_stats['total_generated_samples']}, "
                f"Dropped: {rollouter_stats['dropped_stale_samples']}, "
                f"Errors: {rollouter_stats['generation_errors']}"
            )

            print(
                f"Queue - Size: {queue_stats['queue_size']}, "
                f"Produced: {queue_stats['total_produced']}, "
                f"Consumed: {queue_stats['total_consumed']}"
            )

        except Exception as e:
            print(f"Error getting component statistics: {e}")

    def _check_component_health(self):
        """检查组件健康状态"""
        try:
            # 检查trainer是否仍在运行
            if hasattr(self.components["trainer"], "global_steps"):
                current_steps = self.components["trainer"].global_steps
                # 可以添加更多健康检查逻辑
                print(current_steps)

            # 检查rollouter是否仍在运行
            rollouter_stats = ray.get(self.components["rollouter"].get_statistics.remote(), timeout=5.0)

            if not rollouter_stats["is_running"]:
                print("Rollouter is not running!")
                # 可以尝试重启或报告错误

        except Exception as e:
            print(f"Health check failed: {e}")

    def _cleanup_resources(self):
        """清理资源"""
        print("Cleaning up resources...")

        try:
            # 停止Rollouter
            if "rollouter" in self.components:
                print("Shutting down Rollouter...")
                try:
                    shutdown_future = self.components["rollouter"].shutdown.remote()
                    ray.get(shutdown_future, timeout=10.0)
                except Exception as e:
                    print(f"Error shutting down Rollouter: {e}")

            # 清理MessageQueue
            if "message_queue_client" in self.components:
                print("Cleaning up MessageQueue...")
                try:
                    self.components["message_queue_client"].shutdown()
                except Exception as e:
                    print(f"Error cleaning up MessageQueue: {e}")

            # 清理参数同步器
            if "param_synchronizer" in self.components:
                print("Cleaning up parameter synchronizer...")
                # TODO: 添加参数同步器的清理逻辑

            print("Resource cleanup completed")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_training_status(self) -> dict:
        """获取训练状态"""
        if not self.running or "trainer" not in self.components:
            return {"status": "not_running"}

        try:
            trainer_stats = self.components["trainer"].get_statistics()
            rollouter_stats = ray.get(self.components["rollouter"].get_statistics.remote(), timeout=5.0)

            return {
                "status": "running",
                "trainer_stats": trainer_stats,
                "rollouter_stats": rollouter_stats,
            }
        except Exception as e:
            print(f"Error getting training status: {e}")
            return {"status": "error", "error": str(e)}


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
