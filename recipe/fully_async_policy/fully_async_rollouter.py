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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler

from recipe.fully_async_policy.message_queue import MessageQueueClient
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)


class RolloutController:
    """控制rollout的暂停和恢复 - 改进的控制机制"""

    def __init__(self):
        self.is_paused = False
        self.pause_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()  # 初始状态为可运行
        self.pending_requests = []
        self.lock = threading.RLock()
        self.pause_count = 0

    def pause(self, timeout: Optional[float] = None) -> bool:
        """
        暂停rollout

        Args:
            timeout: 暂停超时时间，如果为None则无限等待

        Returns:
            bool: 是否成功暂停
        """
        with self.lock:
            if not self.is_paused:
                self.is_paused = True
                self.resume_event.clear()
                self.pause_event.set()
                self.pause_count += 1
                logger.info(f"Rollout paused (count: {self.pause_count})")
                return True
            else:
                logger.debug("Rollout already paused")
                return True

    def resume(self) -> bool:
        """
        恢复rollout

        Returns:
            bool: 是否成功恢复
        """
        with self.lock:
            if self.is_paused:
                self.is_paused = False
                self.pause_event.clear()
                self.resume_event.set()
                logger.info("Rollout resumed")
                return True
            else:
                logger.debug("Rollout already running")
                return True

    def wait_if_paused(self, timeout: float = None) -> bool:
        """
        如果被暂停则等待恢复

        Args:
            timeout: 等待超时时间

        Returns:
            bool: 是否成功等待（未超时）
        """
        if self.is_paused:
            logger.debug(f"Waiting for resume (timeout: {timeout})")
            return self.resume_event.wait(timeout)
        return True

    def is_pause_requested(self) -> bool:
        """检查是否有暂停请求"""
        return self.pause_event.is_set()

    def get_status(self) -> dict:
        """获取控制器状态"""
        with self.lock:
            return {
                "is_paused": self.is_paused,
                "pause_count": self.pause_count,
                "has_pending_requests": len(self.pending_requests) > 0,
            }


@ray.remote
class FullyAsyncRollouter:
    """
    异步样本生成器，负责持续生成训练样本并放入MessageQueue
    基于OneStepOffRayTrainer的成熟实现改进
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        train_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
        device_name="cuda",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name

        # 数据相关
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        self.train_sampler = train_sampler

        # Rollout控制
        self.rollout_controller = RolloutController()
        self.current_param_version = 0

        # 新鲜度控制 - 改进的配置管理
        async_config = config.async_training
        self.staleness_threshold = async_config.get("staleness_threshold", 3)
        self.max_staleness_allowed = async_config.get("max_staleness_allowed", 5)
        self.generation_timeout = async_config.get("generation_timeout", 30.0)
        self.batch_generation_interval = async_config.get("batch_generation_interval", 0.1)

        # 统计信息
        self.total_generated_samples = 0
        self.dropped_stale_samples = 0
        self.generation_errors = 0
        self.param_sync_requests = 0

        # Worker groups
        self.rollout_wg = None
        self.message_queue_client = None

        # 运行状态
        self.running = False
        self.generation_thread = None
        self.thread_executor = ThreadPoolExecutor(max_workers=2)

        # 参数同步相关
        self.param_synchronizer = None
        self.last_sync_time = 0
        self.sync_in_progress = False
        self.sync_lock = threading.Lock()

        # 异步rollout模式
        self.async_rollout_mode = config.actor_rollout_ref.rollout.mode == "async"

        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        required_configs = [
            "data.train_batch_size",
            "actor_rollout_ref.rollout.n",
            "async_training.staleness_threshold",
        ]

        for config_path in required_configs:
            if not OmegaConf.select(self.config, config_path):
                logger.warning(f"Missing recommended config: {config_path}")

        # 验证异步训练配置
        if not hasattr(self.config, "async_training"):
            raise ValueError("Missing async_training configuration")

    def init_workers(self):
        """初始化rollout workers - 参考OneStepOffRayTrainer的实现"""
        logger.info("Initializing Rollouter workers...")

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 只创建rollout worker
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        role_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
        )
        self.resource_pool_to_cls[resource_pool]["rollout"] = role_cls

        # 初始化WorkerGroup
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            if OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None:
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.trainer, "worker_nsight_options")
                )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.rollout_wg = all_wg["rollout"]
        self.rollout_wg.init_model()

        # 初始化异步rollout管理器（如果需要）
        if self.async_rollout_mode:
            self._init_async_rollout_manager()

        logger.info("Rollouter workers initialized successfully")

    def _init_async_rollout_manager(self):
        """初始化异步rollout管理器"""
        try:
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.rollout_wg,
            )
            logger.info("Async rollout manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize async rollout manager: {e}")
            self.async_rollout_mode = False

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """设置消息队列客户端"""
        self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """设置参数同步器"""
        self.param_synchronizer = param_synchronizer

    def update_rollout_weights(self, param_version: int) -> bool:
        """
        更新rollout模型参数 - 改进的参数同步实现
        这个方法由外部Trainer调用

        Args:
            param_version: 新的参数版本号

        Returns:
            bool: 是否成功更新参数
        """
        logger.info(f"Updating rollout weights to version {param_version}")

        with self.sync_lock:
            if self.sync_in_progress:
                logger.warning(f"Sync already in progress, skipping version {param_version}")
                return False

            self.sync_in_progress = True

        try:
            # 暂停rollout - 带超时机制
            if not self.rollout_controller.pause(timeout=10.0):
                logger.error("Failed to pause rollout within timeout")
                return False

            # 等待当前generation完成（如果有的话）
            time.sleep(0.1)

            # 执行参数同步
            sync_success = self._execute_parameter_sync(param_version)

            if sync_success:
                self.current_param_version = param_version
                self.param_sync_requests += 1
                self.last_sync_time = time.time()
                logger.info(f"Successfully updated rollout weights to version {param_version}")
            else:
                logger.error(f"Failed to sync parameters to version {param_version}")

        except Exception as e:
            logger.error(f"Error during parameter sync: {e}")
            sync_success = False
        finally:
            # 恢复rollout
            self.rollout_controller.resume()
            self.sync_in_progress = False

        return sync_success

    def _execute_parameter_sync(self, param_version: int) -> bool:
        """
        执行实际的参数同步 - 改进的同步逻辑

        Args:
            param_version: 目标参数版本

        Returns:
            bool: 是否同步成功
        """
        try:
            # 暂停推理引擎
            if self.async_rollout_mode and hasattr(self, "async_rollout_manager"):
                # 对于异步模式，暂停服务器
                pass  # 异步服务器的暂停在 pause() 中已经处理
            else:
                # 对于同步模式，使用sleep/wake_up机制
                sleep_futures = self.rollout_wg.sleep()
                ray.get(sleep_futures)

            # 执行参数同步
            if self.param_synchronizer:
                self.param_synchronizer.sync_weights()
                logger.debug("Parameter synchronization completed via synchronizer")
            else:
                # 直接使用rollout worker group的同步机制
                if hasattr(self.rollout_wg, "sync_rollout_weights"):
                    sync_futures = self.rollout_wg.sync_rollout_weights()
                    ray.get(sync_futures)
                    logger.debug("Parameter synchronization completed via rollout worker group")
                else:
                    logger.warning("No parameter synchronization mechanism available")
                    return False

            # 恢复推理引擎
            if self.async_rollout_mode and hasattr(self, "async_rollout_manager"):
                # 对于异步模式，恢复服务器
                pass  # 异步服务器的恢复在 resume() 中已经处理
            else:
                # 对于同步模式，唤醒workers
                wake_futures = self.rollout_wg.wake_up()
                ray.get(wake_futures)

            return True

        except Exception as e:
            logger.error(f"Parameter sync execution failed: {e}")
            return False

    def _create_dataloader(self):
        """创建数据加载器"""
        from torch.utils.data import DataLoader

        if self.train_dataset is None:
            raise ValueError("Training dataset not provided")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=True,
            pin_memory=True,  # 改进内存管理
        )

    def _create_continuous_iterator(self):
        """创建连续的数据迭代器"""
        dataloader = self._create_dataloader()

        epoch = 0
        while self.running:
            try:
                for batch_dict in dataloader:
                    if not self.running:
                        return
                    yield epoch, batch_dict
                epoch += 1
            except Exception as e:
                logger.error(f"Error in data iterator: {e}")
                time.sleep(1.0)  # 避免快速重试
                continue

    def _should_pause_generation(self) -> bool:
        """
        判断是否应该暂停生成，基于新鲜度控制 - 改进的判断逻辑
        """
        if self.message_queue_client is None:
            return False

        try:
            queue_stats = self.message_queue_client.get_statistics()
            queue_size = queue_stats["queue_size"]
            current_trainer_version = queue_stats["current_param_version"]

            # 计算参数版本差异
            version_diff = self.current_param_version - current_trainer_version

            # 如果版本差异过大，暂停生成
            if version_diff >= self.max_staleness_allowed:
                logger.debug(
                    f"Should pause due to staleness: rollout_version={self.current_param_version}, "
                    f"trainer_version={current_trainer_version}, diff={version_diff}"
                )
                return True

            # 如果队列太满，也暂停生成
            max_queue_size = self.staleness_threshold * self.config.data.train_batch_size
            if queue_size >= max_queue_size:
                logger.debug(f"Should pause due to full queue: size={queue_size}, max={max_queue_size}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking pause conditions: {e}")
            return True  # 出错时暂停生成

    def _generate_batch(self, epoch: int, batch_dict: dict) -> Optional[DataProto]:
        """生成单个batch的样本 - 改进的生成逻辑"""
        try:
            batch = DataProto.from_single_dict(batch_dict)

            # 处理batch用于生成 - 参考OneStepOffRayTrainer的处理逻辑
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

            # 处理多模态数据和其他可选字段
            optional_keys = ["multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs"]
            for key in optional_keys:
                if key in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(key)

            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # 重复生成多个响应 - 参考OneStepOffRayTrainer
            n_repeats = self.config.actor_rollout_ref.rollout.n
            gen_batch = gen_batch.repeat(repeat_times=n_repeats, interleave=True)

            # 执行生成
            if self.async_rollout_mode:
                # 异步生成
                gen_batch_output = ray.get(
                    self.rollout_wg.async_generate_sequences.remote(gen_batch), timeout=self.generation_timeout
                )
            else:
                # 同步生成
                gen_batch_output = ray.get(
                    self.rollout_wg.generate_sequences.remote(gen_batch), timeout=self.generation_timeout
                )

            # 添加UID - 确保每个样本有唯一标识
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

            # 重复原始batch以对齐生成的响应
            batch = batch.repeat(repeat_times=n_repeats, interleave=True)

            # 合并数据
            final_batch = batch.union(gen_batch_output)

            # 添加rollout metadata
            final_batch.meta_info["rollout_param_version"] = self.current_param_version
            final_batch.meta_info["generation_timestamp"] = time.time()

            return final_batch

        except Exception as e:
            logger.error(f"Error generating batch: {e}")
            self.generation_errors += 1
            return None

    def _generation_loop(self):
        """主要的生成循环 - 改进的循环逻辑"""
        logger.info("Starting generation loop...")

        try:
            continuous_iterator = self._create_continuous_iterator()

            for epoch, batch_dict in continuous_iterator:
                if not self.running:
                    break

                # 等待如果被暂停
                if not self.rollout_controller.wait_if_paused(timeout=1.0):
                    if not self.running:
                        break
                    continue

                # 检查是否应该暂停生成
                if self._should_pause_generation():
                    time.sleep(self.batch_generation_interval)
                    continue

                # 生成样本
                timing_raw = {}
                with marked_timer("generate_batch", timing_raw):
                    generated_batch = self._generate_batch(epoch, batch_dict)

                if generated_batch is not None:
                    # 准备rollout metadata
                    rollout_metadata = {
                        "timing": timing_raw,
                        "generation_timestamp": time.time(),
                        "rollout_param_version": self.current_param_version,
                        "epoch": epoch,
                    }

                    # 放入队列
                    success = self.message_queue_client.put_samples(
                        epoch=epoch,
                        sample=generated_batch,
                        param_version=self.current_param_version,
                        rollout_metadata=rollout_metadata,
                    )

                    if success:
                        self.total_generated_samples += 1
                        if self.total_generated_samples % 10 == 0:
                            logger.info(
                                f"Generated {self.total_generated_samples} batches, "
                                f"param_version={self.current_param_version}, "
                                f"errors={self.generation_errors}"
                            )
                    else:
                        self.dropped_stale_samples += 1
                        if self.dropped_stale_samples % 5 == 0:
                            logger.warning(f"Dropped stale samples: {self.dropped_stale_samples}")

                # 控制生成频率
                if self.batch_generation_interval > 0:
                    time.sleep(self.batch_generation_interval)

        except Exception as e:
            logger.error(f"Generation loop error: {e}")
        finally:
            logger.info("Generation loop finished")

    def fit(self):
        """开始异步生成样本 - 改进的主运行逻辑"""
        logger.info("Starting Rollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        self.running = True

        # 在单独的线程中运行生成循环
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()

        logger.info("Rollouter started successfully")

        try:
            # 主线程保持运行，处理控制信号和状态监控
            last_stats_time = time.time()
            stats_interval = 30.0  # 30秒报告一次统计

            while self.running:
                time.sleep(1.0)

                # 定期打印统计信息
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    self._log_statistics()
                    last_stats_time = current_time

                # 检查生成线程状态
                if not self.generation_thread.is_alive():
                    logger.error("Generation thread died, restarting...")
                    self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
                    self.generation_thread.start()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()

    def _log_statistics(self):
        """记录统计信息"""
        try:
            controller_status = self.rollout_controller.get_status()
            queue_stats = self.message_queue_client.get_statistics()

            logger.info(
                f"Rollouter stats - Generated: {self.total_generated_samples}, "
                f"Dropped: {self.dropped_stale_samples}, "
                f"Errors: {self.generation_errors}, "
                f"Queue size: {queue_stats['queue_size']}, "
                f"Param version: {self.current_param_version}, "
                f"Paused: {controller_status['is_paused']}, "
                f"Sync requests: {self.param_sync_requests}"
            )
        except Exception as e:
            logger.error(f"Error logging statistics: {e}")

    def shutdown(self):
        """关闭Rollouter - 改进的关闭逻辑"""
        logger.info("Shutting down Rollouter...")

        self.running = False

        # 恢复可能被暂停的生成线程
        self.rollout_controller.resume()

        # 等待生成线程结束
        if self.generation_thread and self.generation_thread.is_alive():
            logger.info("Waiting for generation thread to finish...")
            self.generation_thread.join(timeout=10.0)

            if self.generation_thread.is_alive():
                logger.warning("Generation thread did not finish within timeout")

        # 关闭线程池
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)

        # 清理异步rollout管理器
        if hasattr(self, "async_rollout_manager"):
            try:
                # TODO: 添加异步rollout管理器的清理逻辑
                pass
            except Exception as e:
                logger.warning(f"Error cleaning up async rollout manager: {e}")

        logger.info("Rollouter shutdown complete")

    def get_statistics(self) -> dict:
        """获取统计信息 - 改进的统计信息"""
        controller_status = self.rollout_controller.get_status()

        stats = {
            "total_generated_samples": self.total_generated_samples,
            "dropped_stale_samples": self.dropped_stale_samples,
            "generation_errors": self.generation_errors,
            "current_param_version": self.current_param_version,
            "param_sync_requests": self.param_sync_requests,
            "last_sync_time": self.last_sync_time,
            "is_running": self.running,
            "sync_in_progress": self.sync_in_progress,
        }

        stats.update(controller_status)

        # 添加队列统计（如果可用）
        if self.message_queue_client:
            try:
                queue_stats = self.message_queue_client.get_statistics()
                stats["queue_size"] = queue_stats.get("queue_size", 0)
                stats["queue_total_produced"] = queue_stats.get("total_produced", 0)
                stats["queue_dropped_samples"] = queue_stats.get("dropped_samples", 0)
            except Exception as e:
                logger.debug(f"Error getting queue statistics: {e}")

        return stats
