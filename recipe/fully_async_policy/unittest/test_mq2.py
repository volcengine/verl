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
from omegaconf import DictConfig

from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient, QueueSample


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


class TestConcurrency:
    """测试并发场景"""

    def setup_method(self):
        """每个测试方法前的设置"""
        if not ray.is_initialized():
            ray.init()

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

    def test_consume_first_produce_later(self, message_queue_client, mock_data_proto):
        """测试先消费、后生产的场景 - 验证阻塞和唤醒机制"""
        consumer_result = []
        producer_result = []
        start_time = time.time()

        def consumer_task():
            """消费者任务：先启动，等待生产者生产数据"""
            # 记录开始消费的时间
            consumer_start = time.time()
            # 这里会阻塞等待，直到有至少2个样本可用
            samples = message_queue_client.get_samples(min_batch_count=3)
            consumer_end = time.time()
            consumer_result.append(
                {
                    "success": True,
                    "samples_count": len(samples),
                    "wait_time": consumer_end - consumer_start,
                    "samples": samples,
                }
            )

        def producer_task():
            """生产者任务：延迟1秒后开始生产"""
            time.sleep(4.0)
            producer_start = time.time()
            message_queue_client.put_sample(
                sample=mock_data_proto,
                param_version=1,
            )
            time.sleep(1)
            message_queue_client.put_sample(
                sample=mock_data_proto,
                param_version=1,
            )
            time.sleep(1)
            message_queue_client.put_sample(
                sample=mock_data_proto,
                param_version=1,
            )
            producer_end = time.time()
            producer_result.append(
                {
                    "put_count": 3,
                    "produce_time": producer_end - producer_start,
                }
            )

            print("produce finish")

        # 启动消费者线程（先启动）
        consumer_thread = threading.Thread(target=consumer_task, name="Consumer")
        time.sleep(3)
        # 启动生产者线程（后启动）
        producer_thread = threading.Thread(target=producer_task, name="Producer")

        consumer_thread.start()
        time.sleep(0.1)  # 确保消费者先开始等待
        producer_thread.start()

        print("=========", flush=True)
        #
        # # 等待两个线程完成（设置超时避免死锁）
        producer_thread.join()
        print("producer_result", producer_result, flush=True)
        consumer_thread.join()
        print("consumer_result", consumer_result, flush=True)

        # 验证结果
        assert len(consumer_result) == 1, "消费者应该执行一次"

        consumer_data = consumer_result[0]
        producer_data = producer_result[0]

        # 验证生产者成功
        assert producer_data["put_count"] == 3, "应该生产2批数据"
        assert consumer_data["samples_count"] == 3, "消费者应该获取到2个样本"

        # 验证队列状态
        final_queue_size = message_queue_client.get_queue_size()
        assert final_queue_size == 0, "队列应该被清空"

        stats = message_queue_client.get_statistics()
        assert stats["total_produced"] == 3, "应该生产了2个样本"
        assert stats["total_consumed"] == 3, "应该消费了2个样本"
        #


# 运行测试的示例配置
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
