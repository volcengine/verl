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
import os
import threading
import time
from unittest.mock import Mock

import pytest
import ray
from recipe.fully_async_policy.message_queue import QueueSample, MessageQueue, MessageQueueClient
from omegaconf import DictConfig


@pytest.fixture
def mock_data_proto():
    """Mock数据对象"""
    return Mock()


@pytest.fixture
def basic_config():
    """基础配置"""
    return DictConfig({"async_training": {"staleness_threshold": 3}})


@pytest.fixture
def queue_config():
    """队列配置"""
    return DictConfig({"async_training": {"staleness_threshold": 2}})


@pytest.fixture
def ray_setup():
    """设置Ray环境"""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def message_queue_client(ray_setup, basic_config):
    """创建MessageQueue actor并返回其客户端"""
    actor = MessageQueue.remote(basic_config, max_queue_size=10)
    client = MessageQueueClient(actor)
    yield client
    client.shutdown()


class TestMessageQueue:
    """测试MessageQueue（通过MessageQueueClient）"""

    def test_put_samples_success(self, message_queue_client, mock_data_proto):
        """测试成功放入samples"""
        samples = [mock_data_proto, mock_data_proto]
        metadata_list = [{"test": "data1"}, {"test": "data2"}]

        result = message_queue_client.put_batch(
            epoch=1, batch=samples, param_version=1, rollout_metadata_list=metadata_list
        )

        assert result is True

        # 检查队列大小
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 2

        # 检查统计信息
        stats = message_queue_client.get_statistics()
        assert stats["total_produced"] == 2
        assert stats["queue_size"] == 2

    def test_put_samples_without_metadata(self, message_queue_client, mock_data_proto):
        """测试不提供metadata时的处理"""
        samples = [mock_data_proto, mock_data_proto]

        result = message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=None)

        assert result is True
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 2

    def test_put_samples_metadata_mismatch(self, message_queue_client, mock_data_proto):
        """测试metadata长度不匹配的处理"""
        samples = [mock_data_proto, mock_data_proto]
        metadata_list = [{"test": "data1"}]  # 长度不匹配

        result = message_queue_client.put_batch(
            epoch=1, batch=samples, param_version=1, rollout_metadata_list=metadata_list
        )

        assert result is False  # 应该失败
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 0

    def test_put_samples_staleness_check(self, message_queue_client, mock_data_proto):
        """测试新鲜度检查"""
        # 更新参数版本为5
        message_queue_client.update_param_version(5)

        # 尝试放入版本过旧的batch（版本差异>=3会被拒绝）
        samples = [mock_data_proto]
        result = message_queue_client.put_batch(
            epoch=1,
            batch=samples,
            param_version=2,  # 5-2=3, 达到阈值
            rollout_metadata_list=None,
        )

        assert result is False

        # 检查统计信息中的丢弃样本数
        stats = message_queue_client.get_statistics()
        assert stats["dropped_samples"] == 1

    def test_put_samples_queue_overflow(self, message_queue_client, mock_data_proto):
        """测试队列溢出处理"""
        # 填满队列（最大容量10）
        for i in range(6):  # 每次放入2个，总共12个，超过最大容量10
            samples = [mock_data_proto, mock_data_proto]
            message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=None)

        # 队列大小应该保持在最大值
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 10

        # 检查统计信息
        stats = message_queue_client.get_statistics()
        assert stats["dropped_samples"] == 2  # 超出的2个被丢弃

    def test_get_samples_success(self, message_queue_client, mock_data_proto):
        """测试成功获取samples"""
        # 先放入一些samples
        samples = [mock_data_proto, mock_data_proto, mock_data_proto]
        metadata_list = [{"index": 0}, {"index": 1}, {"index": 2}]
        message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=metadata_list)

        # 获取2个samples
        retrieved_samples = message_queue_client.get_batch(min_batch_count=2)

        assert retrieved_samples is not None
        assert len(retrieved_samples) == 2
        assert all(isinstance(sample, QueueSample) for sample in retrieved_samples)

        # 检查队列大小减少
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 1

        # 检查统计信息
        stats = message_queue_client.get_statistics()
        assert stats["total_consumed"] == 2

    def test_get_samples_blocking_behavior(self, message_queue_client, mock_data_proto):
        """测试阻塞行为"""
        result = []

        def get_samples():
            # 这会阻塞直到有足够样本
            samples = message_queue_client.get_batch(min_batch_count=2)
            result.append(samples)

        def put_samples_later():
            time.sleep(0.5)  # 延迟放入
            samples = [mock_data_proto, mock_data_proto]
            message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=None)

        # 启动消费者线程
        consumer_thread = threading.Thread(target=get_samples)
        producer_thread = threading.Thread(target=put_samples_later)

        consumer_thread.start()
        producer_thread.start()

        # 等待两个线程完成
        producer_thread.join(timeout=2)
        consumer_thread.join(timeout=2)

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_update_param_version(self, message_queue_client):
        """测试更新参数版本"""
        message_queue_client.update_param_version(10)
        stats = message_queue_client.get_statistics()
        assert stats["current_param_version"] == 10

    def test_clear_queue(self, message_queue_client, mock_data_proto):
        """测试清空队列"""
        # 先添加一些样本
        samples = [mock_data_proto, mock_data_proto, mock_data_proto]
        message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=None)

        # 清空队列
        message_queue_client.clear_queue()

        # 检查队列大小
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 0

    def test_get_queue_size(self, message_queue_client, mock_data_proto):
        """测试获取队列大小"""
        assert message_queue_client.get_queue_size() == 0

        samples = [mock_data_proto]
        message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=None)
        assert message_queue_client.get_queue_size() == 1

    def test_get_statistics(self, message_queue_client):
        """测试获取统计信息"""
        stats = message_queue_client.get_statistics()

        expected_keys = {
            "queue_size",
            "total_produced",
            "total_consumed",
            "dropped_samples",
            "current_param_version",
            "staleness_threshold",
            "max_queue_size",
        }
        assert set(stats.keys()) == expected_keys
        assert isinstance(stats["queue_size"], int)
        assert isinstance(stats["total_produced"], int)
        assert isinstance(stats["total_consumed"], int)

    def test_get_memory_usage(self, message_queue_client, mock_data_proto):
        """测试获取内存使用统计"""
        # 添加一些样本
        samples = [mock_data_proto, mock_data_proto]
        message_queue_client.put_batch(epoch=1, batch=samples, param_version=1, rollout_metadata_list=None)

        memory_stats = message_queue_client.get_memory_usage()

        expected_keys = {"queue_samples", "estimated_memory_bytes", "estimated_memory_mb"}
        assert set(memory_stats.keys()) == expected_keys
        assert memory_stats["queue_samples"] == 2
        assert memory_stats["estimated_memory_bytes"] > 0
        assert memory_stats["estimated_memory_mb"] > 0

    def test_shutdown(self, ray_setup, basic_config):
        """测试关闭功能"""
        # 创建新的actor用于测试关闭
        actor = MessageQueue.remote(basic_config, max_queue_size=10)
        client = MessageQueueClient(actor)

        # 关闭应该不抛出异常
        client.shutdown()


class TestConcurrency:
    """测试并发场景"""

    def setup_method(self):
        """每个测试方法前的设置"""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)

    def teardown_method(self):
        """每个测试方法后的清理"""
        if ray.is_initialized():
            ray.shutdown()

    def create_message_queue_client(self, config=None):
        """创建MessageQueue client的辅助方法"""
        if config is None:
            config = DictConfig({"async_training": {"staleness_threshold": 3}})
        actor = MessageQueue.remote(config, max_queue_size=10)
        return MessageQueueClient(actor)

    def test_concurrent_put_get(self, mock_data_proto):
        """测试并发放入和获取"""
        client = self.create_message_queue_client()
        try:
            results = []

            def producer():
                for i in range(50):
                    samples = [mock_data_proto, mock_data_proto]
                    result = client.put_batch(epoch=i, batch=samples, param_version=1, rollout_metadata_list=None)
                    results.append(("put", result))
                    time.sleep(0.1)

            def consumer():
                for _ in range(100):
                    try:
                        retrieved_samples = client.get_batch(min_batch_count=1)
                        results.append(("get", len(retrieved_samples) > 0))
                    except Exception as e:
                        print(e)
                        results.append(("get", False))
                    time.sleep(0.1)

            # 启动生产者和消费者线程
            producer_thread = threading.Thread(target=producer)
            consumer_thread = threading.Thread(target=consumer)

            producer_thread.start()
            time.sleep(0.05)
            consumer_thread.start()

            producer_thread.join(timeout=5)
            consumer_thread.join(timeout=5)

            # 检查结果
            put_results = [r[1] for r in results if r[0] == "put"]
            get_results = [r[1] for r in results if r[0] == "get"]

            assert all(put_results)
            assert all(get_results)
        finally:
            client.shutdown()


# 运行测试的示例配置
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
