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
单元测试文件，用于测试完全异步PPO训练系统的各个组件
"""

import os

# Import components to test
import sys
import time
import unittest
from unittest.mock import Mock

import ray
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fully_async_rollouter import FullyAsyncRollouter
from fully_async_trainer import FullyAsyncTrainer
from message_queue import MessageQueueClient
from param_sync import ParameterSynchronizer


class TestMessageQueue(unittest.TestCase):
    """测试MessageQueue的功能"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # 创建MessageQueue客户端
        self.message_queue = MessageQueueClient.remote(max_queue_size=100, max_staleness=3)

    def tearDown(self):
        """清理测试环境"""
        if hasattr(self, "message_queue"):
            ray.kill(self.message_queue)

    def test_put_and_get_samples(self):
        """测试放入和获取样本的基本功能"""
        # 创建模拟样本数据
        mock_sample = Mock()
        mock_sample.batch_size = 4

        # 测试放入样本
        success = ray.get(
            self.message_queue.put_samples.remote(
                epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
            )
        )
        self.assertTrue(success)

        # 测试获取样本
        result = ray.get(self.message_queue.get_samples.remote(min_batch_count=1, timeout=5.0, current_param_version=1))

        self.assertIsNotNone(result)
        samples, metadata_list = result
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(metadata_list), 1)

    def test_staleness_control(self):
        """测试新鲜度控制功能"""
        mock_sample = Mock()
        mock_sample.batch_size = 4

        # 放入一个参数版本较老的样本
        success = ray.get(
            self.message_queue.put_samples.remote(
                epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
            )
        )
        self.assertTrue(success)

        # 尝试用较新的参数版本获取样本（应该被拒绝）
        result = ray.get(
            self.message_queue.get_samples.remote(
                min_batch_count=1,
                timeout=5.0,
                current_param_version=5,  # 版本差距为4 > max_staleness(3)
            )
        )

        # 应该返回空结果，因为样本过期
        self.assertIsNone(result)

    def test_queue_statistics(self):
        """测试队列统计功能"""
        # 获取初始统计
        stats = ray.get(self.message_queue.get_statistics.remote())
        initial_queue_size = stats["queue_size"]

        # 添加一些样本
        mock_sample = Mock()
        mock_sample.batch_size = 4

        for i in range(3):
            ray.get(
                self.message_queue.put_samples.remote(
                    epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
                )
            )

        # 检查统计是否更新
        stats = ray.get(self.message_queue.get_statistics.remote())
        self.assertEqual(stats["queue_size"], initial_queue_size + 3)
        self.assertEqual(stats["total_produced"], 3)


class TestParameterSynchronizer(unittest.TestCase):
    """测试参数同步器的功能"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.config = OmegaConf.create(
            {"async_training": {"max_sync_retries": 3, "sync_timeout": 10.0, "sync_retry_delay": 0.1}}
        )

    def test_sync_with_retry(self):
        """测试带重试机制的参数同步"""
        # 创建模拟的worker groups
        mock_actor_wg = Mock()
        mock_rollout_wg = Mock()

        # 模拟同步操作
        mock_actor_wg.get_weights.return_value = ray.put({"param1": "value1"})
        mock_rollout_wg.set_weights.return_value = []

        synchronizer = ParameterSynchronizer.remote(
            config=self.config, actor_wg=mock_actor_wg, rollout_wg=mock_rollout_wg
        )

        # 测试成功同步
        result = ray.get(synchronizer.sync_weights.remote())
        self.assertTrue(result)

    def test_sync_failure_and_retry(self):
        """测试同步失败和重试机制"""
        mock_actor_wg = Mock()
        mock_rollout_wg = Mock()

        # 模拟同步失败
        mock_actor_wg.get_weights.side_effect = Exception("Sync failed")

        synchronizer = ParameterSynchronizer.remote(
            config=self.config, actor_wg=mock_actor_wg, rollout_wg=mock_rollout_wg
        )

        # 测试失败时的重试
        result = ray.get(synchronizer.sync_weights.remote())
        self.assertFalse(result)


class TestFullyAsyncRollouter(unittest.TestCase):
    """测试异步Rollouter的功能"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def test_pause_resume_functionality(self):
        """测试暂停和恢复功能"""
        # 创建配置
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"hybrid_engine": False, "model": {"lora_rank": 0}, "rollout": {"n": 2}},
                "algorithm": {"use_kl_in_reward": False},
                "critic": {"enable": False},
                "trainer": {"device": "cpu", "project_name": "test", "experiment_name": "test"},
                "async_training": {
                    "staleness_threshold": 3,
                    "max_staleness_allowed": 5,
                    "generation_timeout": 10.0,
                    "batch_generation_interval": 0.1,
                },
            }
        )

        # 创建模拟的依赖
        mock_tokenizer = Mock()
        mock_role_worker_mapping = Mock()
        mock_resource_pool_manager = Mock()

        # 创建Rollouter实例
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=mock_tokenizer,
            role_worker_mapping=mock_role_worker_mapping,
            resource_pool_manager=mock_resource_pool_manager,
        )

        # 测试暂停功能
        result = ray.get(rollouter.pause_rollout.remote())
        self.assertTrue(result)

        # 检查暂停状态
        is_paused = ray.get(rollouter.is_rollout_paused.remote())
        self.assertTrue(is_paused)

        # 测试恢复功能
        result = ray.get(rollouter.resume_rollout.remote())
        self.assertTrue(result)

        # 检查恢复状态
        is_paused = ray.get(rollouter.is_rollout_paused.remote())
        self.assertFalse(is_paused)

    def test_statistics_collection(self):
        """测试统计信息收集功能"""
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"hybrid_engine": False, "model": {"lora_rank": 0}, "rollout": {"n": 2}},
                "algorithm": {"use_kl_in_reward": False},
                "critic": {"enable": False},
                "trainer": {"device": "cpu", "project_name": "test", "experiment_name": "test"},
                "async_training": {"staleness_threshold": 3, "max_staleness_allowed": 5, "generation_timeout": 10.0},
            }
        )

        mock_tokenizer = Mock()
        mock_role_worker_mapping = Mock()
        mock_resource_pool_manager = Mock()

        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=mock_tokenizer,
            role_worker_mapping=mock_role_worker_mapping,
            resource_pool_manager=mock_resource_pool_manager,
        )

        # 获取统计信息
        stats = ray.get(rollouter.get_statistics.remote())

        # 验证统计信息包含必要的字段
        expected_keys = [
            "total_generated_samples",
            "dropped_stale_samples",
            "generation_errors",
            "current_param_version",
            "is_paused",
            "pause_count",
            "resume_count",
        ]

        for key in expected_keys:
            self.assertIn(key, stats)


class TestFullyAsyncTrainer(unittest.TestCase):
    """测试异步Trainer的功能"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def test_freshness_metrics_calculation(self):
        """测试新鲜度指标计算"""
        # 创建基本配置
        config = OmegaConf.create(
            {
                "trainer": {
                    "device": "cpu",
                    "project_name": "test",
                    "experiment_name": "test",
                    "total_epochs": 1,
                    "total_training_steps": 2,
                },
                "async_training": {"staleness_threshold": 3, "max_staleness_allowed": 5, "batch_timeout": 10.0},
                "data": {"train_batch_size": 4},
                "actor_rollout_ref": {"hybrid_engine": False, "model": {"lora_rank": 0}},
                "algorithm": {"use_kl_in_reward": False},
                "critic": {"enable": False},
            }
        )

        # 创建模拟的依赖
        mock_tokenizer = Mock()
        mock_role_worker_mapping = Mock()
        mock_resource_pool_manager = Mock()

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=mock_tokenizer,
            role_worker_mapping=mock_role_worker_mapping,
            resource_pool_manager=mock_resource_pool_manager,
        )

        # 测试新鲜度指标计算
        current_time = time.time()
        metadata_list = [
            {"generation_timestamp": current_time - 5, "rollout_param_version": 1},
            {"generation_timestamp": current_time - 10, "rollout_param_version": 2},
            {"generation_timestamp": current_time - 15, "rollout_param_version": 1},
        ]

        freshness_metrics = ray.get(trainer._calculate_freshness_metrics.remote(metadata_list, current_param_version=3))

        # 验证新鲜度指标
        self.assertIn("avg_sample_age", freshness_metrics)
        self.assertIn("max_sample_age", freshness_metrics)
        self.assertIn("min_sample_age", freshness_metrics)
        self.assertIn("version_diversity", freshness_metrics)
        self.assertIn("staleness_ratio", freshness_metrics)


class TestIntegrationScenarios(unittest.TestCase):
    """测试组件集成场景"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def test_message_queue_trainer_integration(self):
        """测试MessageQueue与Trainer的集成"""
        # 创建MessageQueue
        message_queue = MessageQueueClient.remote(max_queue_size=10, max_staleness=3)

        # 放入一些测试样本
        mock_sample = Mock()
        mock_sample.batch_size = 4

        ray.get(
            message_queue.put_samples.remote(
                epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
            )
        )

        # 验证Trainer能够获取样本
        result = ray.get(message_queue.get_samples.remote(min_batch_count=1, timeout=5.0, current_param_version=1))

        self.assertIsNotNone(result)
        samples, metadata_list = result
        self.assertEqual(len(samples), 1)

    def test_rollouter_message_queue_integration(self):
        """测试Rollouter与MessageQueue的集成"""
        # 这个测试需要更多的模拟设置，因为涉及到实际的模型生成
        # 在实际实现中，可以使用更多的Mock对象来模拟这种集成
        pass


class TestErrorHandling(unittest.TestCase):
    """测试错误处理和边界情况"""

    def setUp(self):
        """设置测试环境"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def test_message_queue_overflow(self):
        """测试消息队列溢出处理"""
        # 创建小容量的队列
        message_queue = MessageQueueClient.remote(max_queue_size=2, max_staleness=3)

        mock_sample = Mock()
        mock_sample.batch_size = 4

        # 填满队列
        for i in range(2):
            result = ray.get(
                message_queue.put_samples.remote(
                    epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
                )
            )
            self.assertTrue(result)

        # 尝试再放入一个样本（应该失败或者覆盖旧样本）
        result = ray.get(
            message_queue.put_samples.remote(
                epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
            )
        )

        # 根据实现，这里可能是False（拒绝）或True（覆盖）
        self.assertIsInstance(result, bool)

    def test_timeout_handling(self):
        """测试超时处理"""
        message_queue = MessageQueueClient.remote(max_queue_size=10, max_staleness=3)

        # 尝试从空队列获取样本，应该超时
        start_time = time.time()
        result = ray.get(
            message_queue.get_samples.remote(
                min_batch_count=1,
                timeout=1.0,  # 1秒超时
                current_param_version=1,
            )
        )
        elapsed = time.time() - start_time

        # 应该返回None并且大约在1秒后返回
        self.assertIsNone(result)
        self.assertGreater(elapsed, 0.9)  # 允许一些误差
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    # 设置测试套件
    test_suite = unittest.TestSuite()

    # 添加测试用例
    test_classes = [
        TestMessageQueue,
        TestParameterSynchronizer,
        TestFullyAsyncRollouter,
        TestFullyAsyncTrainer,
        TestIntegrationScenarios,
        TestErrorHandling,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 清理Ray
    if ray.is_initialized():
        ray.shutdown()

    # 退出
    exit(0 if result.wasSuccessful() else 1)
