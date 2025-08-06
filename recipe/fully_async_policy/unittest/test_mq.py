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


class TestMessageQueue:
    """测试MessageQueue（通过MessageQueueClient）"""

    def test_put_samples_success(self, message_queue_client, mock_data_proto):
        """测试成功放入samples"""
        samples = [mock_data_proto, mock_data_proto]
        metadata_list = [{"test": "data1"}, {"test": "data2"}]

        result = message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=metadata_list)

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

        result = message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=None)

        assert result is True
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 2

    def test_put_samples_metadata_mismatch(self, message_queue_client, mock_data_proto):
        """测试metadata长度不匹配的处理"""
        samples = [mock_data_proto, mock_data_proto]
        metadata_list = [{"test": "data1"}]  # 长度不匹配

        result = message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=metadata_list)

        assert result is False  # 应该失败
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 0

    def test_put_samples_staleness_check(self, message_queue_client, mock_data_proto):
        """测试新鲜度检查"""
        # 更新参数版本为5
        message_queue_client.update_param_version(5)

        # 尝试放入版本过旧的batch（版本差异>=3会被拒绝）
        samples = [mock_data_proto]
        result = message_queue_client.put_sample(
            sample=samples,
            param_version=2,  # 5-2=3, 达到阈值
            rollout_metadata=None,
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
            message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=None)

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
        message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=metadata_list)

        # 获取2个samples
        retrieved_samples = message_queue_client.get_samples(min_batch_count=2)

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
            samples = message_queue_client.get_samples(min_batch_count=2)
            result.append(samples)

        def put_samples_later():
            time.sleep(0.5)  # 延迟放入
            samples = [mock_data_proto, mock_data_proto]
            message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=None)

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
        message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=None)

        # 清空队列
        message_queue_client.clear_queue()

        # 检查队列大小
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 0

    def test_get_queue_size(self, message_queue_client, mock_data_proto):
        """测试获取队列大小"""
        assert message_queue_client.get_queue_size() == 0

        samples = [mock_data_proto]
        message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=None)
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
        message_queue_client.put_sample(sample=samples, param_version=1, rollout_metadata=None)

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
                    result = client.put_sample(sample=samples, param_version=1, rollout_metadata=None)
                    results.append(("put", result))
                    time.sleep(0.1)

            def consumer():
                for _ in range(100):
                    try:
                        retrieved_samples = client.get_samples(min_batch_count=1)
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

    def test_consume_first_produce_later(self, message_queue_client, mock_data_proto):
        """测试先消费、后生产的场景 - 验证阻塞和唤醒机制"""
        consumer_result = []
        producer_result = []
        start_time = time.time()

        def consumer_task():
            """消费者任务：先启动，等待生产者生产数据"""
            try:
                # 记录开始消费的时间
                consumer_start = time.time()
                # 这里会阻塞等待，直到有至少2个样本可用
                samples = message_queue_client.get_samples(min_batch_count=2)
                consumer_end = time.time()

                consumer_result.append(
                    {
                        "success": True,
                        "samples_count": len(samples),
                        "wait_time": consumer_end - consumer_start,
                        "samples": samples,
                    }
                )
            except Exception as e:
                consumer_result.append({"success": False, "error": str(e), "wait_time": time.time() - consumer_start})

        def producer_task():
            """生产者任务：延迟1秒后开始生产"""
            try:
                # 延迟1秒，确保消费者先开始等待
                time.sleep(1.0)
                producer_start = time.time()

                # 分两次放入，验证消费者会等到足够的样本数量
                samples_1 = mock_data_proto
                result1 = message_queue_client.put_sample(
                    sample=samples_1, param_version=1, rollout_metadata=[{"batch": "first"}]
                )

                # 短暂延迟后放入第二批
                time.sleep(0.1)
                samples_2 = mock_data_proto
                result2 = message_queue_client.put_sample(
                    sample=samples_2, param_version=1, rollout_metadata=[{"batch": "second"}]
                )

                samples_2 = mock_data_proto
                result3 = message_queue_client.put_sample(
                    sample=samples_2, param_version=1, rollout_metadata=[{"batch": "second"}]
                )

                producer_end = time.time()
                producer_result.append(
                    {
                        "success": result1 and result2,
                        "put_count": 2,
                        "produce_time": producer_end - producer_start,
                        "result1": result1,
                        "result2": result2,
                    }
                )

                print("produce finish")

            except Exception as e:
                producer_result.append({"success": False, "error": str(e)})

        # 启动消费者线程（先启动）
        consumer_thread = threading.Thread(target=consumer_task, name="Consumer")
        # 启动生产者线程（后启动）
        producer_thread = threading.Thread(target=producer_task, name="Producer")

        consumer_thread.start()
        time.sleep(0.1)  # 确保消费者先开始等待
        producer_thread.start()

        print("=========")
        #
        # # 等待两个线程完成（设置超时避免死锁）
        producer_thread.join()
        # print("producer_result", producer_result)
        # consumer_thread.join()
        # print("consumer_thread", consumer_result)
        #
        # total_time = time.time() - start_time
        #
        # # 验证结果
        # assert len(consumer_result) == 1, "消费者应该执行一次"
        #
        # consumer_data = consumer_result[0]
        # producer_data = producer_result[0]
        #
        # # 验证生产者成功
        # assert producer_data['success'], f"生产者失败: {producer_data.get('error', '')}"
        # assert producer_data['put_count'] == 2, "应该生产2批数据"
        #
        # # 验证消费者成功
        # assert consumer_data['success'], f"消费者失败: {consumer_data.get('error', '')}"
        # assert consumer_data['samples_count'] == 2, "消费者应该获取到2个样本"
        #
        # # 验证时序：消费者等待时间应该大于1秒（生产者的延迟时间）
        # assert consumer_data['wait_time'] >= 1.0, f"消费者等待时间应该≥1秒，实际: {consumer_data['wait_time']:.2f}秒"
        #
        # # 验证数据完整性
        # assert all(isinstance(sample, QueueSample) for sample in consumer_data['samples']), "获取的样本应该是QueueSample类型"
        #
        # # 验证队列状态
        # final_queue_size = message_queue_client.get_queue_size()
        # assert final_queue_size == 0, "队列应该被清空"
        #
        # stats = message_queue_client.get_statistics()
        # assert stats['total_produced'] == 2, "应该生产了2个样本"
        # assert stats['total_consumed'] == 2, "应该消费了2个样本"
        #
        # print(f"测试成功完成，总耗时: {total_time:.2f}秒")
        # print(f"消费者等待时间: {consumer_data['wait_time']:.2f}秒")
        # print(f"生产者执行时间: {producer_data['produce_time']:.2f}秒")

    def test_multiple_consumers_single_producer(self, message_queue_client, mock_data_proto):
        """测试多个消费者等待单个生产者的场景"""
        consumer_results = []
        producer_result = []

        def consumer_task(consumer_id):
            """消费者任务"""
            try:
                start_time = time.time()
                samples = message_queue_client.get_samples(min_batch_count=1)
                end_time = time.time()

                consumer_results.append(
                    {
                        "id": consumer_id,
                        "success": True,
                        "samples_count": len(samples),
                        "wait_time": end_time - start_time,
                    }
                )
            except Exception as e:
                consumer_results.append({"id": consumer_id, "success": False, "error": str(e)})

        def producer_task():
            """生产者任务：延迟后批量生产"""
            try:
                time.sleep(1.5)  # 确保所有消费者都在等待

                # 生产3批数据，每批1个样本，供3个消费者消费
                for i in range(3):
                    samples = [mock_data_proto]
                    result = message_queue_client.put_sample(
                        sample=samples, param_version=1, rollout_metadata=[{"batch_id": i}]
                    )
                    producer_result.append(result)
                    time.sleep(0.1)  # 短暂间隔

            except Exception as e:
                producer_result.append(False)

        print("# 启动3个消费者线程")
        # consumer_threads = []
        # for i in range(3):
        #     thread = threading.Thread(target=consumer_task, args=(i,), name=f"Consumer-{i}")
        #     consumer_threads.append(thread)
        #     thread.start()
        #     time.sleep(0.1)  # 错开启动时间
        #
        # # 启动生产者线程
        # producer_thread = threading.Thread(target=producer_task, name="Producer")
        # producer_thread.start()
        #
        # # 等待所有线程完成
        # producer_thread.join(timeout=10)
        # for thread in consumer_threads:
        #     thread.join(timeout=10)
        #
        # # 验证结果
        # assert len(consumer_results) == 3, "应该有3个消费者结果"
        # assert len(producer_result) == 3, "应该生产3批数据"
        #
        # # 验证所有消费者都成功
        # for result in consumer_results:
        #     assert result['success'], f"消费者{result['id']}失败: {result.get('error', '')}"
        #     assert result['samples_count'] == 1, f"消费者{result['id']}应该获取1个样本"
        #     assert result['wait_time'] >= 1.5, f"消费者{result['id']}等待时间应该≥1.5秒"
        #
        # # 验证生产者都成功
        # assert all(producer_result), "所有生产操作都应该成功"
        #
        # # 验证最终状态
        # final_stats = message_queue_client.get_statistics()
        # assert final_stats['total_produced'] == 3, "应该总共生产3个样本"
        # assert final_stats['total_consumed'] == 3, "应该总共消费3个样本"
        # assert final_stats['queue_size'] == 0, "队列应该被清空"

    def test_consumer_timeout_scenario(self, message_queue_client, mock_data_proto):
        """测试消费者超时场景（通过关闭队列来模拟）"""
        consumer_result = []

        def consumer_task():
            """消费者任务：等待样本"""
            try:
                start_time = time.time()
                # 尝试获取样本，但没有生产者会生产数据
                samples = message_queue_client.get_samples(min_batch_count=2)
                end_time = time.time()

                consumer_result.append(
                    {"success": True, "samples_count": len(samples), "wait_time": end_time - start_time}
                )
            except Exception as e:
                consumer_result.append({"success": False, "error": str(e)})

        def shutdown_task():
            """延迟关闭队列，模拟超时场景"""
            time.sleep(2.0)  # 让消费者等待2秒
            message_queue_client.shutdown()

        # 启动消费者和关闭任务
        consumer_thread = threading.Thread(target=consumer_task, name="Consumer")
        shutdown_thread = threading.Thread(target=shutdown_task, name="Shutdown")

        consumer_thread.start()
        time.sleep(0.1)
        shutdown_thread.start()

        # 等待线程完成
        shutdown_thread.join(timeout=5)
        consumer_thread.join(timeout=5)

        # 验证结果
        assert len(consumer_result) == 1, "应该有一个消费者结果"

        result = consumer_result[0]
        # 消费者应该在队列关闭后返回空列表
        if result["success"]:
            assert result["samples_count"] == 0, "关闭后应该返回空样本列表"

        print(f"消费者等待了 {result.get('wait_time', 0):.2f} 秒后退出")

    # 运行测试的示例配置


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
