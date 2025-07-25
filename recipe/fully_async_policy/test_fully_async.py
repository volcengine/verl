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

"""
测试完全异步训练工作流的组件
"""

import logging
import unittest
from unittest.mock import Mock

import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.message_queue import DataProto, MessageQueue, MessageQueueClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMessageQueue(unittest.TestCase):
    """测试MessageQueue组件"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(local_mode=True)

        config = OmegaConf.create(
            {
                "async_training": {
                    "freshness_threshold": 3,
                    "max_staleness_allowed": 5,
                }
            }
        )

        self.message_queue = MessageQueue.remote(config, max_queue_size=100)
        self.client = MessageQueueClient(self.message_queue)

    def tearDown(self):
        """清理测试环境"""
        ray.get(self.message_queue.shutdown.remote())
        if ray.is_initialized():
            ray.shutdown()

    def test_basic_put_get(self):
        """测试基本的put和get操作"""
        # 创建mock数据
        mock_batch = Mock(spec=DataProto)

        # 放入样本
        success = self.client.put_batch(epoch=0, batch=mock_batch, param_version=1, rollout_metadata={"test": "data"})
        self.assertTrue(success)

        # 获取样本
        samples = self.client.get_batch(min_batch_count=1, timeout=5.0)
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].epoch, 0)
        self.assertEqual(samples[0].param_version, 1)

    def test_freshness_control(self):
        """测试新鲜度控制"""
        mock_batch = Mock(spec=DataProto)

        # 更新参数版本
        self.client.update_param_version(10)

        # 尝试放入过期样本
        success = self.client.put_batch(
            epoch=0,
            batch=mock_batch,
            param_version=5,  # 版本差异为5，超过阈值3
            rollout_metadata={},
        )
        self.assertFalse(success)  # 应该被拒绝

    def test_queue_statistics(self):
        """测试队列统计信息"""
        stats = self.client.get_statistics()
        self.assertIn("queue_size", stats)
        self.assertIn("total_produced", stats)
        self.assertIn("total_consumed", stats)
        self.assertIn("dropped_samples", stats)


class TestRollouterComponents(unittest.TestCase):
    """测试Rollouter相关组件"""

    def setUp(self):
        """设置测试环境"""
        from .rollouter import RolloutController

        self.controller = RolloutController()

    def test_rollout_controller(self):
        """测试rollout控制器"""
        # 初始状态应该是运行的
        self.assertFalse(self.controller.is_paused)

        # 测试暂停
        self.controller.pause()
        self.assertTrue(self.controller.is_paused)

        # 测试恢复
        self.controller.resume()
        self.assertFalse(self.controller.is_paused)


class TestParameterSync(unittest.TestCase):
    """测试参数同步组件"""

    def test_async_parameter_synchronizer(self):
        """测试异步参数同步器"""
        from recipe.fully_async_policy.param_sync import AsyncParameterSynchronizer

        config = OmegaConf.create({})
        mock_actor_wg = Mock()
        mock_rollouter_actor = Mock()

        sync = AsyncParameterSynchronizer(config, mock_actor_wg, mock_rollouter_actor)

        self.assertEqual(sync.get_current_version(), 0)


def test_integration():
    """集成测试"""
    logger.info("Starting integration test...")

    if not ray.is_initialized():
        ray.init(local_mode=True)

    try:
        # 测试MessageQueue和客户端的集成
        config = OmegaConf.create(
            {
                "async_training": {
                    "freshness_threshold": 3,
                    "max_staleness_allowed": 5,
                }
            }
        )

        message_queue = MessageQueue.remote(config, max_queue_size=10)
        client = MessageQueueClient(message_queue)

        # 模拟生产者-消费者场景
        mock_batch = Mock(spec=DataProto)

        # 生产样本
        for i in range(5):
            success = client.put_batch(epoch=i, batch=mock_batch, param_version=i, rollout_metadata={"batch_id": i})
            assert success, f"Failed to put batch {i}"

        # 消费样本
        samples = client.get_batch(min_batch_count=3, timeout=10.0)
        assert samples is not None, "Failed to get samples"
        assert len(samples) == 3, f"Expected 3 samples, got {len(samples)}"

        # 检查统计信息
        stats = client.get_statistics()
        assert stats["total_produced"] == 5
        assert stats["total_consumed"] == 3

        logger.info("Integration test passed!")

        # 清理
        ray.get(message_queue.shutdown.remote())

    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    # 运行单元测试
    unittest.main(argv=[""], exit=False, verbosity=2)

    # 运行集成测试
    test_integration()

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)
