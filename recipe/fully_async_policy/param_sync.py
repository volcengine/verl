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

import ray
from ray.util.collective import collective

logger = logging.getLogger(__name__)


class ParameterSynchronizer:
    """
    参数同步器，负责在actor和rollout之间同步模型参数
    """

    def __init__(self, config):
        self.config = config
        self.weights_info = None
        self.sync_group_initialized = False

    def initialize_sync_group(self, actor_workers: list, rollout_workers: list):
        """
        初始化参数同步组

        Args:
            actor_workers: actor worker列表
            rollout_workers: rollout worker列表
        """
        logger.info("Initializing parameter synchronization group...")

        try:
            # 获取actor的权重信息
            if actor_workers:
                self.weights_info = ray.get(actor_workers[0].get_actor_weights_info.remote())[0]

                # 设置rollout的权重信息
                for rollout_worker in rollout_workers:
                    ray.get(rollout_worker.set_actor_weights_info.remote(self.weights_info))

            # 创建actor-rollout通信组
            all_workers = actor_workers + rollout_workers
            collective.create_collective_group(
                all_workers,
                len(all_workers),
                list(range(0, len(all_workers))),
                backend="nccl",
                group_name="actor_rollout",
            )

            self.sync_group_initialized = True
            logger.info("Parameter synchronization group initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize sync group: {e}")
            raise

    def sync_weights(self, actor_workers: list, rollout_workers: list):
        """
        同步权重从actor到rollout

        Args:
            actor_workers: actor worker列表
            rollout_workers: rollout worker列表
        """
        if not self.sync_group_initialized:
            raise RuntimeError("Sync group not initialized. Call initialize_sync_group() first.")

        logger.debug("Synchronizing weights from actor to rollout...")

        try:
            # 同步权重
            sync_futures = []

            # Actor端同步
            for actor_worker in actor_workers:
                future = actor_worker.sync_rollout_weights.remote()
                sync_futures.append(future)

            # Rollout端同步
            for rollout_worker in rollout_workers:
                future = rollout_worker.sync_rollout_weights.remote()
                sync_futures.append(future)

            # 等待所有同步完成
            ray.get(sync_futures)

            logger.debug("Weight synchronization completed")

        except Exception as e:
            logger.error(f"Failed to sync weights: {e}")
            raise


@ray.remote
class ParameterSyncManager:
    """
    Ray Actor形式的参数同步管理器
    """

    def __init__(self, config):
        self.config = config
        self.synchronizer = ParameterSynchronizer(config)
        self.actor_workers = []
        self.rollout_workers = []

    def register_workers(self, actor_workers: list, rollout_workers: list):
        """注册worker"""
        self.actor_workers = actor_workers
        self.rollout_workers = rollout_workers

        # 初始化同步组
        self.synchronizer.initialize_sync_group(actor_workers, rollout_workers)

    def sync_parameters(self):
        """执行参数同步"""
        self.synchronizer.sync_weights(self.actor_workers, self.rollout_workers)
        return True

    def get_weights_info(self):
        """获取权重信息"""
        return self.synchronizer.weights_info


class AsyncParameterSynchronizer:
    """
    异步参数同步器，用于完全异步训练工作流
    """

    def __init__(self, config, actor_wg, rollouter_actor):
        """
        Args:
            config: 配置
            actor_wg: actor worker group
            rollouter_actor: rollouter actor引用
        """
        self.config = config
        self.actor_wg = actor_wg
        self.rollouter_actor = rollouter_actor
        self.current_version = 0

    def sync_to_rollouter(self, new_version: int):
        """
        将actor参数同步到rollouter

        Args:
            new_version: 新的参数版本号
        """
        logger.info(f"Syncing parameters to rollouter, version: {new_version}")

        try:
            # 通知rollouter更新参数
            ray.get(self.rollouter_actor.update_rollout_weights.remote(new_version))

            self.current_version = new_version
            logger.info(f"Parameter sync to rollouter completed, version: {new_version}")

        except Exception as e:
            logger.error(f"Failed to sync parameters to rollouter: {e}")
            raise

    def get_current_version(self) -> int:
        """获取当前参数版本"""
        return self.current_version
