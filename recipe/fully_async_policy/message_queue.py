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
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class QueueSample:
    data: Any
    rollout_metadata: dict[str, Any]


@ray.remote(num_cpus=10, max_concurrency=10)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
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

        # Threading for message handling
        self.running = True

        # thread safe
        self.lock = threading.RLock()
        self.consumer_condition = threading.Condition(self.lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        logger.info(
            f"MessageQueue initialized with max_queue_size={max_queue_size},"
            f"staleness_threshold={self.staleness_threshold}"
        )

    def put_sample(self, sample: Any, param_version: int) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            param_version: Parameter version number

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        with self.lock:
            # Check freshness
            staleness = self.current_param_version - param_version
            if staleness >= self.staleness_threshold:
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
            self.consumer_condition.notify()

            if self.total_produced % 100 == 0:
                logger.debug(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")

            return True

    def get_samples(self, min_batch_count: int = 1) -> list[Any]:
        """
        Get batch samples from the queue, wait until enough samples are available

        Args:
            min_batch_count: Get samples at once when sample count meets min_batch

        Returns:
            List[Any]: List of retrieved samples
        """

        print("get_samples")
        with self.lock:
            while len(self.queue) < min_batch_count and self.running:
                print(f"consumer_condition {len(self.queue)}")
                for data in self.queue:
                    if data is None:
                        return []
                self.consumer_condition.wait()

            # If queue is closed and doesn't have enough samples, return empty list
            if not self.running and len(self.queue) < min_batch_count:
                return []

            # Get specified number of samples
            batch_count = min(min_batch_count, len(self.queue))
            samples = []
            for _ in range(batch_count):
                if self.queue:
                    data = self.queue.popleft()
                    if data is None:
                        return []
                    else:
                        samples.append(data)

            self.total_consumed += len(samples)
            return samples

    def update_param_version(self, version: int):
        """Update current parameter version"""
        with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            logger.debug(f"Parameter version updated from {old_version} to {version}")

    def get_queue_size(self) -> int:
        """Get current queue length"""
        with self.lock:
            return len(self.queue)

    def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
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
        """Clear the queue"""
        with self.lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    def shutdown(self):
        """Shutdown the message queue"""
        with self.lock:
            self.running = False
            # Notify all waiting threads so they can exit
            self.consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        with self.lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    if hasattr(sample.data, "batch") and hasattr(sample.data.batch, "__len__"):
                        # If batch info is available, estimate data size
                        batch_size = len(sample.data.batch)
                        sample_size += batch_size * 1000  # Roughly estimate 1KB per batch entry
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 10000  # Roughly estimate 10KB per sample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }


class MessageQueueClient:
    """MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    def put_sample(self, sample: Any, param_version: int) -> bool:
        """Put batch into queue"""
        return ray.get(self.queue_actor.put_sample.remote(sample, param_version))

    def get_samples(self, min_batch_count: int = 1) -> list[Any]:
        """Get batch from queue, wait until enough samples are available"""
        return ray.get(self.queue_actor.get_samples.remote(min_batch_count))

    def update_param_version(self, version: int):
        """Update parameter version"""
        ray.get(self.queue_actor.update_param_version.remote(version))

    def get_queue_size(self) -> int:
        """Get queue size"""
        return ray.get(self.queue_actor.get_queue_size.remote())

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics"""
        return ray.get(self.queue_actor.get_statistics.remote())

    def clear_queue(self):
        """Clear queue"""
        ray.get(self.queue_actor.clear_queue.remote())

    def shutdown(self):
        """Shutdown queue"""
        ray.get(self.queue_actor.shutdown.remote())

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        return ray.get(self.queue_actor.get_memory_usage.remote())
