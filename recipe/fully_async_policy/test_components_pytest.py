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
Pytest测试文件，用于测试完全异步PPO训练系统的各个组件
"""

import time
from unittest.mock import Mock

import pytest
import ray
from omegaconf import OmegaConf


@pytest.fixture
def ray_setup():
    """Ray初始化fixture"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    yield
    # 测试后不关闭Ray，因为其他测试可能还需要


@pytest.fixture
def basic_config():
    """基本配置fixture"""
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"hybrid_engine": False, "model": {"lora_rank": 0}, "rollout": {"n": 2}},
            "algorithm": {"use_kl_in_reward": False},
            "critic": {"enable": False},
            "trainer": {
                "device": "cpu",
                "project_name": "test",
                "experiment_name": "test",
                "total_epochs": 1,
                "total_training_steps": 2,
            },
            "async_training": {
                "staleness_threshold": 3,
                "max_staleness_allowed": 5,
                "generation_timeout": 10.0,
                "batch_timeout": 5.0,
            },
            "data": {"train_batch_size": 4},
        }
    )


class TestMessageQueue:
    """测试MessageQueue功能"""

    def test_message_queue_creation(self, ray_setup):
        """测试MessageQueue创建"""
        try:
            from message_queue import MessageQueueClient

            queue = MessageQueueClient.remote(max_queue_size=10, max_staleness=3)

            # 测试基本功能
            stats = ray.get(queue.get_statistics.remote())
            assert "queue_size" in stats
            assert stats["queue_size"] == 0

            ray.kill(queue)

        except ImportError:
            pytest.skip("MessageQueue not available")

    def test_queue_put_get(self, ray_setup):
        """测试队列的put/get操作"""
        try:
            from message_queue import MessageQueueClient

            queue = MessageQueueClient.remote(max_queue_size=10, max_staleness=3)

            # 创建模拟样本
            mock_sample = Mock()
            mock_sample.batch_size = 4

            # 测试放入样本
            success = ray.get(
                queue.put_sample.remote(
                    epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
                )
            )
            assert success

            # 测试获取样本
            result = ray.get(queue.get_samples.remote(min_batch_count=1, timeout=2.0, current_param_version=1))
            assert result is not None

            ray.kill(queue)

        except ImportError:
            pytest.skip("MessageQueue not available")


class TestRollouter:
    """测试Rollouter功能"""

    def test_rollouter_pause_resume(self, ray_setup, basic_config):
        """测试Rollouter的暂停恢复功能"""
        try:
            from fully_async_rollouter import FullyAsyncRollouter

            # 创建模拟依赖
            mock_tokenizer = Mock()
            mock_role_worker_mapping = {}
            mock_resource_pool_manager = Mock()

            # 创建Rollouter
            rollouter = FullyAsyncRollouter.remote(
                config=basic_config,
                tokenizer=mock_tokenizer,
                role_worker_mapping=mock_role_worker_mapping,
                resource_pool_manager=mock_resource_pool_manager,
            )

            # 测试暂停
            result = ray.get(rollouter.pause_rollout.remote())
            assert result is True

            # 检查状态
            is_paused = ray.get(rollouter.is_rollout_paused.remote())
            assert is_paused is True

            # 测试恢复
            result = ray.get(rollouter.resume_rollout.remote())
            assert result is True

            # 检查状态
            is_paused = ray.get(rollouter.is_rollout_paused.remote())
            assert is_paused is False

            ray.kill(rollouter)

        except ImportError:
            pytest.skip("FullyAsyncRollouter not available")

    def test_rollouter_statistics(self, ray_setup, basic_config):
        """测试Rollouter统计功能"""
        try:
            from fully_async_rollouter import FullyAsyncRollouter

            mock_tokenizer = Mock()
            mock_role_worker_mapping = {}
            mock_resource_pool_manager = Mock()

            rollouter = FullyAsyncRollouter.remote(
                config=basic_config,
                tokenizer=mock_tokenizer,
                role_worker_mapping=mock_role_worker_mapping,
                resource_pool_manager=mock_resource_pool_manager,
            )

            # 获取统计信息
            stats = ray.get(rollouter.get_statistics.remote())

            # 验证必要字段存在
            required_fields = [
                "total_generated_samples",
                "dropped_stale_samples",
                "generation_errors",
                "current_param_version",
                "is_paused",
                "pause_count",
            ]

            for field in required_fields:
                assert field in stats

            ray.kill(rollouter)

        except ImportError:
            pytest.skip("FullyAsyncRollouter not available")


class TestTrainer:
    """测试Trainer功能"""

    def test_trainer_creation(self, ray_setup, basic_config):
        """测试Trainer创建"""
        try:
            from fully_async_trainer import FullyAsyncTrainer

            mock_tokenizer = Mock()
            mock_role_worker_mapping = {}
            mock_resource_pool_manager = Mock()

            trainer = FullyAsyncTrainer.remote(
                config=basic_config,
                tokenizer=mock_tokenizer,
                role_worker_mapping=mock_role_worker_mapping,
                resource_pool_manager=mock_resource_pool_manager,
            )

            # 基本验证
            assert trainer is not None

            ray.kill(trainer)

        except ImportError:
            pytest.skip("FullyAsyncTrainer not available")


class TestParameterSync:
    """测试参数同步功能"""

    def test_param_sync_creation(self, ray_setup):
        """测试参数同步器创建"""
        try:
            from param_sync import ParameterSynchronizer

            config = OmegaConf.create(
                {"async_training": {"max_sync_retries": 3, "sync_timeout": 10.0, "sync_retry_delay": 0.1}}
            )

            mock_actor_wg = Mock()
            mock_rollout_wg = Mock()

            synchronizer = ParameterSynchronizer.remote(
                config=config, actor_wg=mock_actor_wg, rollout_wg=mock_rollout_wg
            )

            assert synchronizer is not None

            ray.kill(synchronizer)

        except ImportError:
            pytest.skip("ParameterSynchronizer not available")


class TestIntegration:
    """集成测试"""

    def test_basic_workflow_simulation(self, ray_setup):
        """测试基本工作流模拟"""
        # 这是一个简化的集成测试，模拟基本的工作流
        try:
            from message_queue import MessageQueueClient

            # 创建消息队列
            queue = MessageQueueClient.remote(max_queue_size=5, max_staleness=2)

            # 模拟生产者（Rollouter）
            mock_sample = Mock()
            mock_sample.batch_size = 2

            # 放入样本
            success = ray.get(
                queue.put_sample.remote(
                    epoch=1, sample=mock_sample, param_version=1, rollout_metadata={"timestamp": time.time()}
                )
            )
            assert success

            # 模拟消费者（Trainer）
            result = ray.get(queue.get_samples.remote(min_batch_count=1, timeout=2.0, current_param_version=1))
            assert result is not None

            samples, metadata_list = result
            assert len(samples) == 1
            assert len(metadata_list) == 1

            ray.kill(queue)

        except ImportError:
            pytest.skip("Integration test components not available")


class TestErrorHandling:
    """错误处理测试"""

    def test_timeout_handling(self, ray_setup):
        """测试超时处理"""
        try:
            from message_queue import MessageQueueClient

            queue = MessageQueueClient.remote(max_queue_size=5, max_staleness=2)

            # 测试从空队列超时获取
            start_time = time.time()
            result = ray.get(
                queue.get_samples.remote(
                    min_batch_count=1,
                    timeout=1.0,  # 1秒超时
                    current_param_version=1,
                )
            )
            elapsed = time.time() - start_time

            assert result is None
            assert 0.9 <= elapsed <= 2.0  # 允许一些误差

            ray.kill(queue)

        except ImportError:
            pytest.skip("MessageQueue not available")


if __name__ == "__main__":
    # 如果直接运行此文件，执行所有测试
    pytest.main([__file__, "-v"])
