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
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class QueueSample:
    """单个batch样本，包含参数版本和新鲜度信息"""

    id: str
    data: Any
    param_version: int
    timestamp: float
    rollout_metadata: dict[str, Any]


@ray.remote(num_cpus=10, max_concurrency=10)
class MessageQueue:
    """
    简化的Ray-based异步消息队列，用于Rollouter和Trainer之间的通信
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        self.max_queue_size = max_queue_size
        self.queue = deque(maxlen=max_queue_size)
        self.current_param_version = 0

        # 安全地获取配置值
        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.staleness_threshold = getattr(config.async_training, "staleness_threshold", 3)
            else:
                self.staleness_threshold = 3
        except (AttributeError, RecursionError):
            self.staleness_threshold = 3

        # Threading for message handling
        self.running = True

        # 线程安全
        self.lock = threading.RLock()
        self.consumer_condition = threading.Condition(self.lock)

        # 统计信息
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        logger.info(
            f"MessageQueue initialized with max_queue_size={max_queue_size},"
            "staleness_threshold={self.staleness_threshold}"
        )

    def put_samples(
            self, samples: list[Any] | Any, param_version: int, rollout_metadata: dict[str, Any] = None
    ) -> bool:
        """
        放入一个batch样本到队列

        Args:
            samples: 样本数据
            param_version: 参数版本号
            rollout_metadata: rollout相关的元数据

        Returns:
            bool: 是否成功放入队列
        """
        with self.lock:
            # 检查新鲜度
            staleness = self.current_param_version - param_version
            if staleness >= self.staleness_threshold:
                self.dropped_samples += 1
                logger.debug(f"Dropped stale sample: staleness={staleness}, threshold={self.staleness_threshold}")
                return False

            for sample in samples:
                queue_sample = QueueSample(
                    id=str(uuid.uuid4()),
                    data=sample,
                    param_version=param_version,
                    timestamp=time.time(),
                    rollout_metadata=rollout_metadata or {},
                )

                # 如果队列满了，移除最旧的样本，一般不会发生
                if len(self.queue) >= self.max_queue_size:
                    removed = self.queue.popleft()
                    self.dropped_samples += 1
                    logger.warning(f"Queue full, dropped sample {removed.id}")

                self.queue.append(queue_sample)
                self.total_produced += 1

            # 通知等待的消费者
            self.consumer_condition.notify()

            if self.total_produced % 100 == 0:
                logger.debug(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")

            return True

    def get_samples(self, min_batch: int = 1) -> list[QueueSample]:
        """
        从队列获取batch样本，一直等待直到有足够样本

        Args:
            min_batch: sample数量满足min_batch，一次性获取

        Returns:
            List[QueueSample]: 获取的样本列表
        """
        with self.lock:
            while len(self.queue) < min_batch and self.running:
                self.consumer_condition.wait()

            # 如果队列已关闭且没有足够样本，返回空列表
            if not self.running and len(self.queue) < min_batch:
                return []

            # 获取指定数量的样本
            batch_count = min(min_batch, len(self.queue))
            samples = []
            for _ in range(batch_count):
                if self.queue:
                    samples.append(self.queue.popleft())

            self.total_consumed += len(samples)
            return samples

    def update_param_version(self, version: int):
        """更新当前参数版本"""
        with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            logger.debug(f"Parameter version updated from {old_version} to {version}")

    def get_queue_size(self) -> int:
        """获取当前队列长度"""
        with self.lock:
            return len(self.queue)

    def get_statistics(self) -> dict[str, Any]:
        """获取队列统计信息"""
        with self.lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "current_param_version": self.current_param_version,
                "staleness_threshold": self.staleness_threshold,
                "max_queue_size": self.max_queue_size,
            }

    def clear_queue(self):
        """清空队列"""
        with self.lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    def shutdown(self):
        """关闭消息队列"""
        with self.lock:  # 修正：需要加锁
            self.running = False
            # 通知所有等待的线程，让它们能够退出
            self.consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    def get_memory_usage(self) -> dict:
        """获取内存使用统计"""
        with self.lock:
            # 估算队列中样本的内存使用
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # 估算单个样本的大小（简化估算）
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    if hasattr(sample.data, "batch") and hasattr(sample.data.batch, "__len__"):
                        # 如果有batch信息，估算数据大小
                        batch_size = len(sample.data.batch)
                        sample_size += batch_size * 1000  # 粗略估算每个batch条目1KB
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 10000  # 粗略估算每个样本10KB

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }


class MessageQueueClient:
    """MessageQueue的客户端，用于与MessageQueue Actor通信"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    def put_samples(
            self, samples: list[Any], param_version: int, rollout_metadata_list: list[dict[str, Any]] = None
    ) -> bool:
        """放入batch到队列"""
        return ray.get(self.queue_actor.put_samples.remote(samples, param_version, rollout_metadata_list))

    def get_samples(self, min_batch_count: int = 1) -> list[QueueSample]:
        """从队列获取batch，一直等待直到有足够样本"""
        return ray.get(self.queue_actor.get_samples.remote(min_batch_count))

    def update_param_version(self, version: int):
        """更新参数版本"""
        ray.get(self.queue_actor.update_param_version.remote(version))

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return ray.get(self.queue_actor.get_queue_size.remote())

    def get_statistics(self) -> dict[str, Any]:
        """获取统计信息"""
        return ray.get(self.queue_actor.get_statistics.remote())

    def clear_queue(self):
        """清空队列"""
        ray.get(self.queue_actor.clear_queue.remote())

    def shutdown(self):
        """关闭队列"""
        ray.get(self.queue_actor.shutdown.remote())

    def get_memory_usage(self) -> dict:
        """获取内存使用统计"""
        return ray.get(self.queue_actor.get_memory_usage.remote())
