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

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import ray
import zmq
from filelock import FileLock
from omegaconf import DictConfig

from verl import DataProto


@dataclass
class BatchSample:
    """单个batch样本，包含参数版本和新鲜度信息"""

    batch_id: str
    epoch: int
    data: DataProto
    param_version: int
    timestamp: float
    rollout_metadata: dict[str, Any]


@ray.remote(num_cpus=1)
class MessageQueue:
    """
    基于ZeroMQ的异步消息队列，用于Rollouter和Trainer之间的通信
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        self.max_queue_size = max_queue_size
        self.queue = deque(maxlen=max_queue_size)
        self.current_param_version = 0
        self.freshness_threshold = config.async_training.get("freshness_threshold", 3)

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = None
        self.address = None
        self._setup_zmq()

        # Threading for message handling
        self.running = True
        self.lock = threading.RLock()
        self.consumer_waiting = False
        self.consumer_condition = threading.Condition(self.lock)

        # Statistics
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

    def _setup_zmq(self):
        """设置ZeroMQ socket"""
        with FileLock("/tmp/verl_message_queue.lock"):
            # 使用TCP socket
            import socket as sock

            with sock.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

            self.address = f"tcp://127.0.0.1:{port}"
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.bind(self.address)

    def put_batch(
        self, epoch: int, batch: DataProto, param_version: int, rollout_metadata: dict[str, Any] = None
    ) -> bool:
        """
        放入一个batch样本到队列

        Args:
            epoch: 当前epoch
            batch: 样本数据
            param_version: 参数版本号
            rollout_metadata: rollout相关的元数据

        Returns:
            bool: 是否成功放入队列
        """
        with self.lock:
            # 检查新鲜度
            staleness = self.current_param_version - param_version
            if staleness >= self.freshness_threshold:
                self.dropped_samples += 1
                return False

            sample = BatchSample(
                batch_id=str(uuid.uuid4()),
                epoch=epoch,
                data=batch,
                param_version=param_version,
                timestamp=time.time(),
                rollout_metadata=rollout_metadata or {},
            )

            # 如果队列满了，移除最旧的样本
            if len(self.queue) >= self.max_queue_size:
                removed = self.queue.popleft()
                self.dropped_samples += 1
                print(f"Queue full, dropped sample {removed.batch_id}")

            self.queue.append(sample)
            self.total_produced += 1

            # 通知等待的消费者
            if self.consumer_waiting:
                self.consumer_condition.notify()

            return True

    def get_batch(self, min_batch_count: int = 1, timeout: float = 30.0) -> Optional[list[BatchSample]]:
        """
        从队列获取batch样本

        Args:
            min_batch_count: 最小batch数量
            timeout: 超时时间（秒）

        Returns:
            Optional[List[BatchSample]]: 获取的样本列表，如果超时返回None
        """
        with self.lock:
            start_time = time.time()

            while len(self.queue) < min_batch_count:
                if time.time() - start_time > timeout:
                    return None

                self.consumer_waiting = True
                self.consumer_condition.wait(timeout=1.0)
                self.consumer_waiting = False

            # 获取指定数量的样本
            batch_count = min(min_batch_count, len(self.queue))
            samples = []
            for _ in range(batch_count):
                if self.queue:
                    samples.append(self.queue.popleft())

            self.total_consumed += len(samples)
            return samples

    def update_param_version(self, version: int):
        """更新当前参数版本"""
        with self.lock:
            self.current_param_version = version

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
                "freshness_threshold": self.freshness_threshold,
            }

    def clear_queue(self):
        """清空队列"""
        with self.lock:
            self.queue.clear()

    def shutdown(self):
        """关闭消息队列"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

    def get_address(self) -> str:
        """获取ZeroMQ地址"""
        return self.address


class MessageQueueClient:
    """MessageQueue的客户端，用于与MessageQueue Actor通信"""

    def __init__(self, queue_actor: ray.ActorHandle):
        self.queue_actor = queue_actor

    def put_batch(
        self, epoch: int, batch: DataProto, param_version: int, rollout_metadata: dict[str, Any] = None
    ) -> bool:
        """放入batch到队列"""
        return ray.get(self.queue_actor.put_batch.remote(epoch, batch, param_version, rollout_metadata))

    def get_batch(self, min_batch_count: int = 1, timeout: float = 30.0) -> Optional[list[BatchSample]]:
        """从队列获取batch"""
        return ray.get(self.queue_actor.get_batch.remote(min_batch_count, timeout))

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
