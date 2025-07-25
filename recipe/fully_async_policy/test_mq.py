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
from unittest.mock import Mock

import pytest
import ray
from message_queue import BatchSample, MessageQueue, MessageQueueClient
from omegaconf import DictConfig


@pytest.fixture
def mock_data_proto():
    """Mock DataProto对象"""
    return Mock()


@pytest.fixture
def basic_config():
    """基础配置"""
    return DictConfig({"async_training": {"freshness_threshold": 3}})


@pytest.fixture
def queue_config():
    """队列配置"""
    return DictConfig({"async_training": {"freshness_threshold": 2}})


class TestBatchSample:
    """测试BatchSample数据类"""

    def test_batch_sample_creation(self, mock_data_proto):
        """测试BatchSample创建"""
        sample = BatchSample(
            batch_id="test-123",
            epoch=1,
            data=mock_data_proto,
            param_version=5,
            timestamp=1234567890.0,
            rollout_metadata={"key": "value"},
        )

        assert sample.batch_id == "test-123"
        assert sample.epoch == 1
        assert sample.data == mock_data_proto
        assert sample.param_version == 5
        assert sample.timestamp == 1234567890.0
        assert sample.rollout_metadata == {"key": "value"}


class TestMessageQueue:
    """测试MessageQueue类（需要在非Ray环境下测试内部逻辑）"""

    def test_message_queue_init(self, basic_config):
        """测试MessageQueue初始化"""
        # 直接创建MessageQueue实例（不使用Ray装饰器）
        queue = MessageQueue.__ray_actor_class__(basic_config, max_queue_size=100)

        # 确保ZeroMQ初始化成功
        assert queue.context is not None
        assert queue.socket is not None

        # 基本属性检查
        assert queue.max_queue_size == 100
        assert queue.current_param_version == 0
        assert queue.freshness_threshold == 3
        assert len(queue.queue) == 0
        assert queue.total_produced == 0
        assert queue.total_consumed == 0
        assert queue.dropped_samples == 0

        # 清理资源
        queue.shutdown()


@pytest.fixture
def ray_setup():
    """设置Ray环境"""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def message_queue_actor(ray_setup, basic_config):
    """创建MessageQueue actor"""
    actor = MessageQueue.remote(basic_config, max_queue_size=10)
    yield actor
    ray.get(actor.shutdown.remote())


class TestMessageQueueActor:
    """测试MessageQueue Actor"""

    def test_put_batch_success(self, message_queue_actor, mock_data_proto):
        """测试成功放入batch"""
        result = ray.get(
            message_queue_actor.put_batch.remote(
                epoch=1, batch=mock_data_proto, param_version=1, rollout_metadata={"test": "data"}
            )
        )

        assert result is True

        # 检查队列大小
        queue_size = ray.get(message_queue_actor.get_queue_size.remote())
        assert queue_size == 1

        # 检查统计信息
        stats = ray.get(message_queue_actor.get_statistics.remote())
        assert stats["total_produced"] == 1
        assert stats["queue_size"] == 1

    def test_put_batch_staleness_check(self, message_queue_actor, mock_data_proto):
        """测试新鲜度检查"""
        # 更新参数版本为5
        ray.get(message_queue_actor.update_param_version.remote(5))

        # 尝试放入版本过旧的batch（版本差异>=3会被拒绝）
        result = ray.get(
            message_queue_actor.put_batch.remote(
                epoch=1,
                batch=mock_data_proto,
                param_version=2,  # 5-2=3, 达到阈值
                rollout_metadata={},
            )
        )

        assert result is False

        # 检查统计信息中的丢弃样本数
        stats = ray.get(message_queue_actor.get_statistics.remote())
        assert stats["dropped_samples"] == 1

    def test_put_batch_queue_overflow(self, message_queue_actor, mock_data_proto):
        """测试队列溢出处理"""
        # 填满队列（最大容量10）
        for i in range(12):  # 超过最大容量
            ray.get(
                message_queue_actor.put_batch.remote(
                    epoch=1, batch=mock_data_proto, param_version=1, rollout_metadata={}
                )
            )

        # 队列大小应该保持在最大值
        queue_size = ray.get(message_queue_actor.get_queue_size.remote())
        assert queue_size == 10

        # 检查统计信息
        stats = ray.get(message_queue_actor.get_statistics.remote())
        assert stats["dropped_samples"] == 2  # 超出的2个被丢弃

    def test_get_batch_success(self, message_queue_actor, mock_data_proto):
        """测试成功获取batch"""
        # 先放入一些batch
        for i in range(3):
            ray.get(
                message_queue_actor.put_batch.remote(
                    epoch=i, batch=mock_data_proto, param_version=1, rollout_metadata={"index": i}
                )
            )

        # 获取2个batch
        samples = ray.get(message_queue_actor.get_batch.remote(min_batch_count=2, timeout=5.0))

        assert samples is not None
        assert len(samples) == 2
        assert all(isinstance(sample, BatchSample) for sample in samples)

        # 检查队列大小减少
        queue_size = ray.get(message_queue_actor.get_queue_size.remote())
        assert queue_size == 1

        # 检查统计信息
        stats = ray.get(message_queue_actor.get_statistics.remote())
        assert stats["total_consumed"] == 2

    def test_get_batch_timeout(self, message_queue_actor):
        """测试获取batch超时"""
        # 空队列情况下获取batch应该超时
        samples = ray.get(message_queue_actor.get_batch.remote(min_batch_count=1, timeout=1.0))
        assert samples is None

    def test_update_param_version(self, message_queue_actor):
        """测试更新参数版本"""
        ray.get(message_queue_actor.update_param_version.remote(10))

        stats = ray.get(message_queue_actor.get_statistics.remote())
        assert stats["current_param_version"] == 10

    def test_clear_queue(self, message_queue_actor, mock_data_proto):
        """测试清空队列"""
        # 先添加一些样本
        for i in range(3):
            ray.get(message_queue_actor.put_batch.remote(epoch=i, batch=mock_data_proto, param_version=1))

        # 清空队列
        ray.get(message_queue_actor.clear_queue.remote())

        # 检查队列大小
        queue_size = ray.get(message_queue_actor.get_queue_size.remote())
        assert queue_size == 0

    def test_get_statistics(self, message_queue_actor):
        """测试获取统计信息"""
        stats = ray.get(message_queue_actor.get_statistics.remote())

        expected_keys = {
            "queue_size",
            "total_produced",
            "total_consumed",
            "dropped_samples",
            "current_param_version",
            "freshness_threshold",
        }
        assert set(stats.keys()) == expected_keys
        assert isinstance(stats["queue_size"], int)
        assert isinstance(stats["total_produced"], int)
        assert isinstance(stats["total_consumed"], int)


class TestMessageQueueClient:
    """测试MessageQueueClient"""

    def test_client_put_batch(self, message_queue_actor, mock_data_proto):
        """测试客户端放入batch"""
        client = MessageQueueClient(message_queue_actor)

        result = client.put_batch(epoch=1, batch=mock_data_proto, param_version=1, rollout_metadata={"test": "client"})

        assert result is True
        assert client.get_queue_size() == 1

    def test_client_get_batch(self, message_queue_actor, mock_data_proto):
        """测试客户端获取batch"""
        client = MessageQueueClient(message_queue_actor)

        # 先放入一个batch
        client.put_batch(epoch=1, batch=mock_data_proto, param_version=1)

        # 获取batch
        samples = client.get_batch(min_batch_count=1, timeout=5.0)

        assert samples is not None
        assert len(samples) == 1
        assert isinstance(samples[0], BatchSample)

    def test_client_update_param_version(self, message_queue_actor):
        """测试客户端更新参数版本"""
        client = MessageQueueClient(message_queue_actor)

        client.update_param_version(15)

        stats = client.get_statistics()
        assert stats["current_param_version"] == 15

    def test_client_get_queue_size(self, message_queue_actor, mock_data_proto):
        """测试客户端获取队列大小"""
        client = MessageQueueClient(message_queue_actor)

        assert client.get_queue_size() == 0

        client.put_batch(epoch=1, batch=mock_data_proto, param_version=1)
        assert client.get_queue_size() == 1

    def test_client_clear_queue(self, message_queue_actor, mock_data_proto):
        """测试客户端清空队列"""
        client = MessageQueueClient(message_queue_actor)

        # 添加样本
        client.put_batch(epoch=1, batch=mock_data_proto, param_version=1)
        assert client.get_queue_size() == 1

        # 清空队列
        client.clear_queue()
        assert client.get_queue_size() == 0

    def test_client_shutdown(self, message_queue_actor):
        """测试客户端关闭"""
        client = MessageQueueClient(message_queue_actor)

        # 关闭不应该抛出异常
        client.shutdown()


class TestConcurrency:
    """测试并发场景"""

    def test_concurrent_put_get(self, message_queue_actor, mock_data_proto):
        """测试并发放入和获取"""
        client = MessageQueueClient(message_queue_actor)
        results = []

        def producer():
            for i in range(5):
                result = client.put_batch(epoch=i, batch=mock_data_proto, param_version=1)
                results.append(("put", result))
                time.sleep(0.1)

        def consumer():
            for _ in range(3):
                samples = client.get_batch(min_batch_count=1, timeout=2.0)
                results.append(("get", samples is not None))
                time.sleep(0.1)

        # 启动生产者和消费者线程
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        time.sleep(0.05)  # 让生产者先开始
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        # 检查结果
        put_results = [r[1] for r in results if r[0] == "put"]
        get_results = [r[1] for r in results if r[0] == "get"]

        assert all(put_results)  # 所有放入操作都应该成功
        assert all(get_results)  # 所有获取操作都应该成功


# 运行测试的示例配置
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
