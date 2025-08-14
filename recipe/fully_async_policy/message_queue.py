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

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information (preserved from _prepare_generate_batch)
    original_batch_dict: dict[str, Any]

    # AgentLoopOutput from generation
    agent_loop_output: Any  # AgentLoopOutput

    # Metadata
    sample_id: str
    epoch: int
    rollout_n_index: int  # Index within the rollout.n repetitions (0, 1, ..., n-1)
    original_sample_index: int  # Index of the original sample before repetition

    # Processing metadata
    processing_time: float
    generation_timestamp: float
    param_version: int


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    使用 asyncio 实现异步消息队列
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        self.max_queue_size = max_queue_size
        self.queue = deque(maxlen=max_queue_size)
        self.current_param_version = 0

        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.staleness_threshold = getattr(config.async_training, "staleness_threshold", 3)
            else:
                self.staleness_threshold = 3
        except (AttributeError, RecursionError):
            self.staleness_threshold = 3

        # Asyncio for message handling
        self.running = True

        # async safe - 在第一次使用时初始化
        self._lock = None
        self._consumer_condition = None

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        logger.info(
            f"MessageQueue initialized with max_queue_size={max_queue_size},"
            f"staleness_threshold={self.staleness_threshold}"
        )

    async def _ensure_async_primitives(self):
        """确保异步原语已初始化"""
        if self._lock is None:
            self._lock = asyncio.Lock()
            self._consumer_condition = asyncio.Condition(self._lock)

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            param_version: Parameter version number

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        await self._ensure_async_primitives()

        async with self._lock:
            # Check freshness
            staleness = self.current_param_version - param_version
            if staleness > self.staleness_threshold:
                self.dropped_samples += 1
                logger.debug(f"Dropped stale sample: staleness={staleness}, threshold={self.staleness_threshold}")
                return False

            # If queue is full, remove the oldest sample (rarely happens)
            if len(self.queue) >= self.max_queue_size:
                removed = self.queue.popleft()
                self.dropped_samples += 1
                logger.warning(f"Queue full, dropped sample {removed}")
            self.queue.append(sample)
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify()

            if self.total_produced % 100 == 0:
                logger.debug(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")

            return True

    async def get_samples(self, min_batch_count: int = 1) -> tuple[list[Any], int]:
        """
        Get batch samples from the queue, wait until enough samples are available

        Args:
            min_batch_count: Get samples at once when sample count meets min_batch

        Returns:
            List[Any]: List of retrieved samples
        """
        await self._ensure_async_primitives()

        async with self._lock:
            while len(self.queue) < min_batch_count and self.running:
                print(f"[MessageQueue] consumer_condition {len(self.queue)}")
                if len(self.queue) > 0 and self.queue[-1] is None:
                    return [], len(self.queue)
                await self._consumer_condition.wait()

            # If queue is closed and doesn't have enough samples, return empty list
            if not self.running and len(self.queue) < min_batch_count:
                return [], len(self.queue)

            # Get specified number of samples
            batch_count = min(min_batch_count, len(self.queue))
            samples = []
            for _ in range(batch_count):
                if self.queue:
                    data = self.queue.popleft()
                    if data is None:
                        return [], len(self.queue)
                    else:
                        samples.append(data)

            self.total_consumed += len(samples)
            return samples, len(self.queue)

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        await self._ensure_async_primitives()

        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            return data

    async def update_param_version(self, version: int):
        """Update current parameter version"""
        await self._ensure_async_primitives()

        async with self._lock:
            old_version = self.current_param_version
            self.current_param_version = version
            logger.debug(f"Parameter version updated from {old_version} to {version}")

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        await self._ensure_async_primitives()

        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        await self._ensure_async_primitives()

        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "current_param_version": self.current_param_version,
                "staleness_threshold": self.staleness_threshold,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        """Clear the queue"""
        await self._ensure_async_primitives()

        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        await self._ensure_async_primitives()

        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        await self._ensure_async_primitives()

        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample, param_version)
        return await asyncio.wrap_future(future.future())

    async def get_samples(self, min_batch_count: int = 1) -> tuple[list[Any], int]:
        """Get batch from queue, wait until enough samples are available (async)"""
        future = self.queue_actor.get_samples.remote(min_batch_count)
        return await asyncio.wrap_future(future.future())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def update_param_version(self, version: int):
        """Update parameter version (async)"""
        future = self.queue_actor.update_param_version.remote(version)
        await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    # 为了兼容性，保留同步版本的方法（但标记为deprecated）
    def put_sample_sync(self, sample: Any, param_version: int) -> bool:
        """Put batch into queue (sync - deprecated, use put_sample instead)"""
        return ray.get(self.queue_actor.put_sample.remote(sample, param_version))

    def get_samples_sync(self, min_batch_count: int = 1) -> tuple[list[Any], int]:
        """Get batch from queue (sync - deprecated, use get_samples instead)"""
        return ray.get(self.queue_actor.get_samples.remote(min_batch_count))

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())
