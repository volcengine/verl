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
    """控制rollout的暂停和恢复"""

    def __init__(self):
        self.is_paused = False
        self.pause_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()  # 初始状态为可运行
        self.pending_requests = []
        self.lock = threading.RLock()

    def pause(self):
        """暂停rollout"""
        with self.lock:
            if not self.is_paused:
                self.is_paused = True
                self.resume_event.clear()
                self.pause_event.set()
                logger.info("Rollout paused")

    def resume(self):
        """恢复rollout"""
        with self.lock:
            if self.is_paused:
                self.is_paused = False
                self.pause_event.clear()
                self.resume_event.set()
                logger.info("Rollout resumed")

    def wait_if_paused(self, timeout: float = None):
        """如果被暂停则等待恢复"""
        if self.is_paused:
            self.resume_event.wait(timeout)

    def is_pause_requested(self) -> bool:
        """检查是否有暂停请求"""
        return self.pause_event.is_set()


class Rollouter:
    """
    异步样本生成器，负责持续生成训练样本并放入MessageQueue
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
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

        # 新鲜度控制
        self.freshness_threshold = config.async_training.get("freshness_threshold", 3)
        self.max_staleness_allowed = config.async_training.get("max_staleness_allowed", 5)

        # 统计信息
        self.total_generated_samples = 0
        self.dropped_stale_samples = 0
        self.pause_count = 0

        # Worker groups
        self.rollout_wg = None
        self.message_queue_client = None

        # 运行状态
        self.running = False
        self.generation_thread = None

    def init_workers(self):
        """初始化rollout workers"""
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
        logger.info("Rollouter workers initialized successfully")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """设置消息队列客户端"""
        self.message_queue_client = message_queue_client

    def update_rollout_weights(self, param_version: int):
        """
        更新rollout模型参数
        这个方法由外部Trainer调用
        """
        logger.info(f"Updating rollout weights to version {param_version}")

        # 暂停rollout
        self.rollout_controller.pause()

        try:
            # 暂停推理引擎
            ray.get(self.rollout_wg.sleep.remote())

            # 执行参数同步
            # 这里需要与actor建立同步机制
            if hasattr(self, "param_synchronizer") and self.param_synchronizer:
                self.param_synchronizer.sync_weights()
            else:
                logger.warning("Parameter synchronizer not available, skipping weight sync")

            # 更新参数版本
            self.current_param_version = param_version

            # 恢复推理引擎
            ray.get(self.rollout_wg.wake_up.remote())

        finally:
            # 恢复rollout
            self.rollout_controller.resume()

        logger.info(f"Rollout weights updated to version {param_version}")

    def set_parameter_synchronizer(self, param_synchronizer):
        """设置参数同步器"""
        self.param_synchronizer = param_synchronizer

    def _create_dataloader(self):
        """创建数据加载器"""
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.get("dataloader_num_workers", 0),
            drop_last=True,
        )

    def _create_continuous_iterator(self):
        """创建连续的数据迭代器"""
        dataloader = self._create_dataloader()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in dataloader:
                yield epoch, batch_dict

    def _should_pause_generation(self) -> bool:
        """
        判断是否应该暂停生成，基于新鲜度控制
        """
        if self.message_queue_client is None:
            return False

        queue_stats = self.message_queue_client.get_statistics()
        queue_size = queue_stats["queue_size"]
        current_trainer_version = queue_stats["current_param_version"]

        # 计算参数版本差异
        version_diff = self.current_param_version - current_trainer_version

        # 如果版本差异过大，暂停生成
        if version_diff >= self.max_staleness_allowed:
            logger.info(
                f"Pausing generation due to staleness: rollout_version={self.current_param_version}, "
                f"trainer_version={current_trainer_version}, diff={version_diff}"
            )
            return True

        # 如果队列太满，也暂停生成
        max_queue_size = self.freshness_threshold * self.config.data.train_batch_size
        if queue_size >= max_queue_size:
            logger.info(f"Pausing generation due to full queue: size={queue_size}, max={max_queue_size}")
            return True

        return False

    def _generate_batch(self, epoch: int, batch_dict: dict) -> Optional[DataProto]:
        """生成单个batch的样本"""
        try:
            batch = DataProto.from_single_dict(batch_dict)

            # 处理batch用于生成
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

            # 处理多模态数据
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")

            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # 重复生成多个响应
            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

            # 执行生成
            if self.config.actor_rollout_ref.rollout.mode == "async":
                gen_batch_output = ray.get(self.rollout_wg.async_generate_sequences.remote(gen_batch))
            else:
                gen_batch_output = ray.get(self.rollout_wg.generate_sequences.remote(gen_batch))

            # 添加UID
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

            # 重复原始batch以对齐生成的响应
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

            # 合并数据
            final_batch = batch.union(gen_batch_output)

            return final_batch

        except Exception as e:
            logger.error(f"Error generating batch: {e}")
            return None

    def _generation_loop(self):
        """主要的生成循环"""
        logger.info("Starting generation loop...")

        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            if not self.running:
                break

            # 等待如果被暂停
            self.rollout_controller.wait_if_paused(timeout=1.0)

            if not self.running:
                break

            # 检查是否应该暂停生成
            if self._should_pause_generation():
                time.sleep(1.0)  # 等待一段时间再检查
                continue

            # 生成样本
            timing_raw = {}
            with marked_timer("generate_batch", timing_raw):
                generated_batch = self._generate_batch(epoch, batch_dict)

            if generated_batch is not None:
                # 放入队列
                rollout_metadata = {
                    "timing": timing_raw,
                    "generation_timestamp": time.time(),
                }

                success = self.message_queue_client.put_batch(
                    epoch=epoch,
                    batch=generated_batch,
                    param_version=self.current_param_version,
                    rollout_metadata=rollout_metadata,
                )

                if success:
                    self.total_generated_samples += 1
                    if self.total_generated_samples % 10 == 0:
                        logger.info(
                            f"Generated {self.total_generated_samples} batches, "
                            f"param_version={self.current_param_version}"
                        )
                else:
                    self.dropped_stale_samples += 1
                    logger.warning(f"Dropped stale sample, total dropped: {self.dropped_stale_samples}")

        logger.info("Generation loop finished")

    def fit(self):
        """开始异步生成样本"""
        logger.info("Starting Rollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        self.running = True

        # 在单独的线程中运行生成循环
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()

        try:
            # 主线程保持运行，处理控制信号
            while self.running:
                time.sleep(1.0)

                # 定期打印统计信息
                if self.total_generated_samples > 0 and self.total_generated_samples % 100 == 0:
                    queue_stats = self.message_queue_client.get_statistics()
                    logger.info(
                        f"Rollouter stats - Generated: {self.total_generated_samples}, "
                        f"Dropped: {self.dropped_stale_samples}, "
                        f"Queue size: {queue_stats['queue_size']}, "
                        f"Param version: {self.current_param_version}"
                    )

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        """关闭Rollouter"""
        logger.info("Shutting down Rollouter...")

        self.running = False

        # 恢复可能被暂停的生成线程
        self.rollout_controller.resume()

        # 等待生成线程结束
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5.0)

        logger.info("Rollouter shutdown complete")

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_generated_samples": self.total_generated_samples,
            "dropped_stale_samples": self.dropped_stale_samples,
            "current_param_version": self.current_param_version,
            "pause_count": self.pause_count,
            "is_running": self.running,
            "is_paused": self.rollout_controller.is_paused,
        }
