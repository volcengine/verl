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
import time

import ray
from ray.util.collective import collective

logger = logging.getLogger(__name__)


class ParameterSynchronizer:
    """
    参数同步器，负责在actor和rollout之间同步模型参数
    改进版本，具有更好的错误处理和重试机制
    """

    def __init__(self, config):
        self.config = config
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"

        # 同步配置
        self.max_sync_retries = config.async_training.get("max_sync_retries", 3)
        self.sync_timeout = config.async_training.get("sync_timeout", 30.0)
        self.retry_delay = config.async_training.get("sync_retry_delay", 1.0)

        # 统计信息
        self.sync_count = 0
        self.sync_failures = 0
        self.last_sync_time = 0

    def initialize_sync_group(self, actor_workers: list, rollout_workers: list) -> bool:
        """
        初始化参数同步组

        Args:
            actor_workers: actor worker列表
            rollout_workers: rollout worker列表

        Returns:
            bool: 是否成功初始化
        """
        logger.info("Initializing parameter synchronization group...")

        try:
            # 验证workers
            if not actor_workers:
                raise ValueError("No actor workers provided")
            if not rollout_workers:
                raise ValueError("No rollout workers provided")

            # 获取actor的权重信息
            logger.debug("Getting actor weights info...")
            weights_info_future = actor_workers[0].get_actor_weights_info.remote()
            self.weights_info = ray.get(weights_info_future, timeout=10.0)[0]

            if not self.weights_info:
                raise ValueError("Failed to get actor weights info")

            # 设置rollout的权重信息
            logger.debug("Setting rollout weights info...")
            set_weights_futures = []
            for rollout_worker in rollout_workers:
                future = rollout_worker.set_actor_weights_info.remote(self.weights_info)
                set_weights_futures.append(future)

            ray.get(set_weights_futures, timeout=10.0)

            # 创建actor-rollout通信组
            logger.debug("Creating collective communication group...")
            all_workers = actor_workers + rollout_workers

            # 清理可能存在的旧组
            try:
                collective.destroy_collective_group(self.sync_group_name)
            except Exception:
                pass  # 忽略清理错误

            collective.create_collective_group(
                all_workers,
                len(all_workers),
                list(range(0, len(all_workers))),
                backend="nccl",
                group_name=self.sync_group_name,
            )

            self.sync_group_initialized = True
            logger.info("Parameter synchronization group initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize sync group: {e}")
            self.sync_group_initialized = False
            return False

    def sync_weights(self, actor_workers: list, rollout_workers: list) -> bool:
        """
        同步权重从actor到rollout - 改进版本，具有重试和错误处理

        Args:
            actor_workers: actor worker列表
            rollout_workers: rollout worker列表

        Returns:
            bool: 是否同步成功
        """
        if not self.sync_group_initialized:
            logger.error("Sync group not initialized. Call initialize_sync_group() first.")
            return False

        logger.debug("Starting weight synchronization...")
        start_time = time.time()

        for attempt in range(self.max_sync_retries):
            try:
                # 执行同步
                success = self._execute_sync(actor_workers, rollout_workers)

                if success:
                    self.sync_count += 1
                    self.last_sync_time = time.time()
                    sync_duration = self.last_sync_time - start_time
                    logger.debug(f"Weight synchronization completed in {sync_duration:.2f}s")
                    return True
                else:
                    logger.warning(f"Sync attempt {attempt + 1} failed")

            except Exception as e:
                logger.warning(f"Sync attempt {attempt + 1} failed with error: {e}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_sync_retries - 1:
                logger.info(f"Retrying sync in {self.retry_delay}s...")
                time.sleep(self.retry_delay)

        # 所有重试都失败
        self.sync_failures += 1
        logger.error(f"All sync attempts failed. Total failures: {self.sync_failures}")
        return False

    def _execute_sync(self, actor_workers: list, rollout_workers: list) -> bool:
        """
        执行实际的同步操作

        Args:
            actor_workers: actor worker列表
            rollout_workers: rollout worker列表

        Returns:
            bool: 是否同步成功
        """
        try:
            sync_futures = []

            # Actor端同步
            for actor_worker in actor_workers:
                future = actor_worker.sync_rollout_weights.remote()
                sync_futures.append(future)

            # Rollout端同步
            for rollout_worker in rollout_workers:
                future = rollout_worker.sync_rollout_weights.remote()
                sync_futures.append(future)

            # 等待所有同步完成，带超时
            ray.get(sync_futures, timeout=self.sync_timeout)
            return True

        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            return False

    def cleanup(self):
        """清理同步组"""
        if self.sync_group_initialized:
            try:
                collective.destroy_collective_group(self.sync_group_name)
                logger.info("Sync group cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up sync group: {e}")
            finally:
                self.sync_group_initialized = False

    def get_statistics(self) -> dict:
        """获取同步统计信息"""
        return {
            "sync_count": self.sync_count,
            "sync_failures": self.sync_failures,
            "last_sync_time": self.last_sync_time,
            "sync_group_initialized": self.sync_group_initialized,
        }


@ray.remote
class ParameterSyncManager:
    """
    Ray Actor形式的参数同步管理器 - 改进版本
    """

    def __init__(self, config):
        self.config = config
        self.synchronizer = ParameterSynchronizer(config)
        self.actor_workers = []
        self.rollout_workers = []
        self.is_ready = False

    def register_workers(self, actor_workers: list, rollout_workers: list) -> bool:
        """
        注册worker

        Args:
            actor_workers: actor worker列表
            rollout_workers: rollout worker列表

        Returns:
            bool: 是否成功注册
        """
        try:
            self.actor_workers = actor_workers
            self.rollout_workers = rollout_workers

            # 初始化同步组
            success = self.synchronizer.initialize_sync_group(actor_workers, rollout_workers)
            self.is_ready = success

            if success:
                logger.info("ParameterSyncManager ready")
            else:
                logger.error("ParameterSyncManager initialization failed")

            return success
        except Exception as e:
            logger.error(f"Failed to register workers: {e}")
            return False

    def sync_parameters(self) -> bool:
        """
        执行参数同步

        Returns:
            bool: 是否同步成功
        """
        if not self.is_ready:
            logger.error("SyncManager not ready. Call register_workers() first.")
            return False

        return self.synchronizer.sync_weights(self.actor_workers, self.rollout_workers)

    def get_weights_info(self):
        """获取权重信息"""
        return self.synchronizer.weights_info

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = self.synchronizer.get_statistics()
        stats["is_ready"] = self.is_ready
        return stats

    def cleanup(self):
        """清理资源"""
        self.synchronizer.cleanup()
        self.is_ready = False


class AsyncParameterSynchronizer:
    """
    异步参数同步器，用于完全异步训练工作流 - 改进版本
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

        # 同步配置
        self.sync_timeout = config.async_training.get("sync_timeout", 30.0)
        self.max_sync_retries = config.async_training.get("max_sync_retries", 3)
        self.retry_delay = config.async_training.get("sync_retry_delay", 1.0)

        # 统计信息
        self.sync_count = 0
        self.sync_failures = 0
        self.last_sync_time = 0

        # 初始化同步组
        self._init_sync_group()

    def _init_sync_group(self):
        """初始化同步组"""
        try:
            # 获取actor权重信息
            weights_info = self.actor_wg.get_actor_weights_info()[0]

            # 通知rollouter设置权重信息
            ray.get(self.rollouter_actor.set_weights_info.remote(weights_info), timeout=10.0)

            # 创建同步通信组
            actor_workers = self.actor_wg.workers
            rollout_workers = ray.get(self.rollouter_actor.get_rollout_workers.remote(), timeout=10.0)

            all_workers = actor_workers + rollout_workers
            collective.create_collective_group(
                all_workers,
                len(all_workers),
                list(range(0, len(all_workers))),
                backend="nccl",
                group_name="async_actor_rollout",
            )

            logger.info("Async parameter synchronizer initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize async sync group: {e}")

    def sync_to_rollouter(self, new_version: int) -> bool:
        """
        将actor参数同步到rollouter - 改进版本，具有重试机制

        Args:
            new_version: 新的参数版本号

        Returns:
            bool: 是否同步成功
        """
        logger.info(f"Syncing parameters to rollouter, version: {new_version}")
        start_time = time.time()

        for attempt in range(self.max_sync_retries):
            try:
                # 首先同步actor到rollout worker group
                self.actor_wg.sync_rollout_weights()

                # 然后通知rollouter更新参数版本
                sync_future = self.rollouter_actor.update_rollout_weights.remote(new_version)
                sync_result = ray.get(sync_future, timeout=self.sync_timeout)

                if sync_result:
                    self.current_version = new_version
                    self.sync_count += 1
                    self.last_sync_time = time.time()
                    sync_duration = self.last_sync_time - start_time
                    logger.info(f"Parameter sync completed in {sync_duration:.2f}s, version: {new_version}")
                    return True
                else:
                    logger.warning(f"Rollouter rejected sync for version {new_version}")

            except Exception as e:
                logger.warning(f"Sync attempt {attempt + 1} failed: {e}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_sync_retries - 1:
                logger.info(f"Retrying sync in {self.retry_delay}s...")
                time.sleep(self.retry_delay)

        # 所有重试都失败
        self.sync_failures += 1
        logger.error(f"Failed to sync parameters to rollouter after {self.max_sync_retries} attempts")
        return False

    def get_current_version(self) -> int:
        """获取当前参数版本"""
        return self.current_version

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "current_version": self.current_version,
            "sync_count": self.sync_count,
            "sync_failures": self.sync_failures,
            "last_sync_time": self.last_sync_time,
        }
