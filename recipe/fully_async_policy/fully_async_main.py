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
import threading
import time

import hydra
import ray

from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from recipe.fully_async_policy.rollouter import Rollouter
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager

from .fully_async_trainer import FullyAsyncTrainer

logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@ray.remote
class RollouterActor:
    """Rollouter的Ray Actor包装器"""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        processor=None,
        train_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name="cuda",
    ):
        self.rollouter = Rollouter(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

    def init_workers(self):
        """初始化worker"""
        return self.rollouter.init_workers()

    def set_message_queue_client(self, message_queue_client):
        """设置消息队列客户端"""
        return self.rollouter.set_message_queue_client(message_queue_client)

    def set_parameter_synchronizer(self, param_synchronizer):
        """设置参数同步器"""
        return self.rollouter.set_parameter_synchronizer(param_synchronizer)

    def update_rollout_weights(self, param_version: int):
        """更新rollout权重"""
        return self.rollouter.update_rollout_weights(param_version)

    def fit(self):
        """开始生成循环"""
        return self.rollouter.fit()

    def shutdown(self):
        """关闭rollouter"""
        return self.rollouter.shutdown()

    def get_statistics(self):
        """获取统计信息"""
        return self.rollouter.get_statistics()


def run_fully_async_ppo(config):
    """运行完全异步的PPO训练"""
    setup_logging()

    logger.info("Starting fully async PPO training...")

    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            address=os.environ.get("RAY_ADDRESS", None),
            runtime_env={"env_vars": {"NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}},
        )

    try:
        # 创建数据集和采样器
        logger.info("Creating dataset and sampler...")
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
        processor = hf_processor(config.actor_rollout_ref.model.path)

        train_dataset, val_dataset = create_rl_dataset(config, tokenizer, processor)
        train_sampler = create_rl_sampler(config, train_dataset)

        # 创建collate function
        from verl.trainer.ppo.ray_trainer import default_collate_fn

        collate_fn = default_collate_fn

        # 创建奖励函数
        reward_fn, val_reward_fn = load_reward_manager(config, tokenizer)

        # 创建资源池管理器和worker映射
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import (
            Role,
            create_resource_pool_manager,
            create_role_worker_mapping,
        )

        # resource_pool_manager = create_resource_pool_manager(config)
        role_worker_mapping = create_role_worker_mapping(config)

        # 1. 创建MessageQueue
        logger.info("Creating MessageQueue...")
        max_queue_size = config.async_training.get("max_queue_size", 1000)
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)

        # 2. 创建Rollouter Actor
        logger.info("Creating Rollouter...")
        rollouter_actor = RollouterActor.remote(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping={Role.Rollout: role_worker_mapping[Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=RayWorkerGroup,
            processor=processor,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        # 初始化Rollouter
        ray.get(rollouter_actor.init_workers.remote())
        ray.get(rollouter_actor.set_message_queue_client.remote(message_queue_client))

        # 3. 创建Trainer
        logger.info("Creating FullyAsyncTrainer...")
        trainer_role_mapping = {
            role: worker_cls for role, worker_cls in role_worker_mapping.items() if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=RayWorkerGroup,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        # 初始化Trainer
        trainer.init_workers()
        trainer.set_message_queue_client(message_queue_client)
        trainer.set_rollouter_actor(rollouter_actor)

        # 4. 设置参数同步
        logger.info("Setting up parameter synchronization...")
        # param_synchronizer = AsyncParameterSynchronizer(
        #     config=config, actor_wg=trainer.actor_wg, rollouter_actor=rollouter_actor
        # )

        # 5. 启动Rollouter（在后台线程中）
        logger.info("Starting Rollouter in background...")

        def run_rollouter():
            try:
                ray.get(rollouter_actor.fit.remote())
            except Exception as e:
                logger.error(f"Rollouter error: {e}")

        rollouter_thread = threading.Thread(target=run_rollouter, daemon=True)
        rollouter_thread.start()

        # 等待一下让Rollouter启动
        time.sleep(5)

        # 6. 启动Trainer（主线程）
        logger.info("Starting FullyAsyncTrainer...")
        trainer.fit()

        # 7. 关闭
        logger.info("Shutting down...")
        ray.get(rollouter_actor.shutdown.remote())

        # 等待Rollouter线程结束
        rollouter_thread.join(timeout=10)

        # 关闭MessageQueue
        ray.get(message_queue.shutdown.remote())

        logger.info("Fully async PPO training completed successfully!")

    except Exception as e:
        logger.error(f"Error in fully async PPO training: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


@hydra.main(config_path="../one_step_off_policy/config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    """主入口函数"""
    run_fully_async_ppo(config)


if __name__ == "__main__":
    main()
