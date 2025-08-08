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

from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient


@pytest.fixture
def mock_sample():
    """Mock sample data object"""
    return Mock()


@pytest.fixture
def basic_config():
    """Basic configuration"""
    return DictConfig({"async_training": {"staleness_threshold": 3}})


@pytest.fixture
def queue_config():
    """Queue configuration with different staleness threshold"""
    return DictConfig({"async_training": {"staleness_threshold": 2}})


@pytest.fixture
def ray_setup():
    """Setup Ray environment"""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def message_queue_client(ray_setup, basic_config):
    """Create MessageQueue actor and return its client"""
    actor = MessageQueue.remote(basic_config, max_queue_size=10)
    client = MessageQueueClient(actor)
    yield client
    client.shutdown()


class TestMessageQueue:
    """Test MessageQueue (through MessageQueueClient)"""

    def test_put_sample_success(self, message_queue_client, mock_sample):
        """Test successfully putting a sample"""
        result = message_queue_client.put_sample(sample=mock_sample, param_version=1)
        assert result is True

        # Check queue size
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 1

        # Check statistics
        stats = message_queue_client.get_statistics()
        assert stats["total_produced"] == 1
        assert stats["queue_size"] == 1

    def test_put_multiple_samples(self, message_queue_client, mock_sample):
        """Test putting multiple samples"""
        for i in range(3):
            result = message_queue_client.put_sample(sample=mock_sample, param_version=1)
            assert result is True

        # Check queue size
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 3

        # Check statistics
        stats = message_queue_client.get_statistics()
        assert stats["total_produced"] == 3
        assert stats["queue_size"] == 3

    def test_put_sample_staleness_check(self, message_queue_client, mock_sample):
        """Test freshness check when putting samples"""
        # Update parameter version to 5
        message_queue_client.update_param_version(5)

        # Try to put a stale sample (version difference >= 3 will be rejected)
        result = message_queue_client.put_sample(
            sample=mock_sample,
            param_version=2,  # 5-2=3, reaches threshold
        )

        assert result is False

        # Check dropped samples count in statistics
        stats = message_queue_client.get_statistics()
        assert stats["dropped_samples"] == 1

    def test_put_sample_queue_overflow(self, message_queue_client, mock_sample):
        """Test queue overflow handling"""
        # Fill the queue (max capacity 10)
        for i in range(12):  # Put 12 samples, exceeding max capacity 10
            message_queue_client.put_sample(sample=mock_sample, param_version=1)

        # Queue size should stay at maximum value
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 10

        # Check statistics
        stats = message_queue_client.get_statistics()
        assert stats["dropped_samples"] == 2  # 2 samples should be dropped

    def test_get_samples_success(self, message_queue_client, mock_sample):
        """Test successfully getting samples"""
        # First put some samples
        for i in range(3):
            message_queue_client.put_sample(sample=mock_sample, param_version=1)

        # Get 2 samples
        retrieved_samples = message_queue_client.get_samples(min_batch_count=2)

        assert retrieved_samples is not None
        assert len(retrieved_samples) == 2

        # Check queue size decreased
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 1

        # Check statistics
        stats = message_queue_client.get_statistics()
        assert stats["total_consumed"] == 2

    def test_get_samples_blocking_behavior(self, message_queue_client, mock_sample):
        """Test blocking behavior"""
        result = []

        def get_samples():
            # This will block until enough samples are available
            samples = message_queue_client.get_samples(min_batch_count=2)
            result.append(samples)

        def put_samples_later():
            time.sleep(0.5)  # Delay putting samples
            message_queue_client.put_sample(sample=mock_sample, param_version=1)
            message_queue_client.put_sample(sample=mock_sample, param_version=1)

        # Start consumer thread
        consumer_thread = threading.Thread(target=get_samples)
        producer_thread = threading.Thread(target=put_samples_later)

        consumer_thread.start()
        producer_thread.start()

        # Wait for both threads to complete
        producer_thread.join(timeout=2)
        consumer_thread.join(timeout=2)

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_update_param_version(self, message_queue_client):
        """Test updating parameter version"""
        message_queue_client.update_param_version(10)
        stats = message_queue_client.get_statistics()
        assert stats["current_param_version"] == 10

    def test_clear_queue(self, message_queue_client, mock_sample):
        """Test clearing the queue"""
        # First add some samples
        for i in range(3):
            message_queue_client.put_sample(sample=mock_sample, param_version=1)

        # Clear the queue
        message_queue_client.clear_queue()

        # Check queue size
        queue_size = message_queue_client.get_queue_size()
        assert queue_size == 0

    def test_get_queue_size(self, message_queue_client, mock_sample):
        """Test getting queue size"""
        assert message_queue_client.get_queue_size() == 0

        message_queue_client.put_sample(sample=mock_sample, param_version=1)
        assert message_queue_client.get_queue_size() == 1

    def test_get_statistics(self, message_queue_client):
        """Test getting statistics"""
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

    def test_get_memory_usage(self, message_queue_client, mock_sample):
        """Test getting memory usage statistics"""
        # Add some samples
        for i in range(2):
            message_queue_client.put_sample(sample=mock_sample, param_version=1)

        memory_stats = message_queue_client.get_memory_usage()

        expected_keys = {"queue_samples", "estimated_memory_bytes", "estimated_memory_mb"}
        assert set(memory_stats.keys()) == expected_keys
        assert memory_stats["queue_samples"] == 2
        assert memory_stats["estimated_memory_bytes"] > 0
        assert memory_stats["estimated_memory_mb"] > 0

    def test_shutdown(self, ray_setup, basic_config):
        """Test shutdown functionality"""
        # Create new actor for testing shutdown
        actor = MessageQueue.remote(basic_config, max_queue_size=10)
        client = MessageQueueClient(actor)

        # Shutdown should not throw exceptions
        client.shutdown()


class TestConcurrency:
    """Test concurrent scenarios"""

    def setup_method(self):
        """Setup before each test method"""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)

    def teardown_method(self):
        """Cleanup after each test method"""
        if ray.is_initialized():
            ray.shutdown()

    def create_message_queue_client(self, config=None):
        """Helper method to create MessageQueue client"""
        if config is None:
            config = DictConfig({"async_training": {"staleness_threshold": 3}})
        actor = MessageQueue.remote(config, max_queue_size=10)
        return MessageQueueClient(actor)

    def test_concurrent_put_get(self, mock_sample):
        """Test concurrent put and get"""
        client = self.create_message_queue_client()
        try:
            results = []

            def producer():
                for i in range(50):
                    samples = [mock_sample, mock_sample]
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

            # Start producer and consumer threads
            producer_thread = threading.Thread(target=producer)
            consumer_thread = threading.Thread(target=consumer)

            producer_thread.start()
            time.sleep(0.05)
            consumer_thread.start()

            producer_thread.join(timeout=5)
            consumer_thread.join(timeout=5)

            # Check results
            put_results = [r[1] for r in results if r[0] == "put"]
            get_results = [r[1] for r in results if r[0] == "get"]

            assert all(put_results)
            assert all(get_results)
        finally:
            client.shutdown()

    def test_consume_first_produce_later(self, message_queue_client, mock_data_proto):
        """Test consume first, produce later scenario - verify blocking and wake-up mechanism"""
        consumer_result = []
        producer_result = []

        def consumer_task():
            """Consumer task: start first, wait for producer to generate data"""
            # Record the start time of consumption
            consumer_start = time.time()
            # This will block until at least 3 samples are available
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
            """Producer task: start producing after a delay"""
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

        # Start consumer thread (first)
        consumer_thread = threading.Thread(target=consumer_task, name="Consumer")
        time.sleep(3)
        # Start producer thread (later)
        producer_thread = threading.Thread(target=producer_task, name="Producer")

        consumer_thread.start()
        time.sleep(0.1)
        producer_thread.start()

        print("=========", flush=True)

        producer_thread.join()
        print("producer_result", producer_result, flush=True)
        consumer_thread.join()
        print("consumer_result", consumer_result, flush=True)

        assert len(consumer_result) == 1, "消费者应该执行一次"

        consumer_data = consumer_result[0]
        producer_data = producer_result[0]

        assert producer_data["put_count"] == 3
        assert consumer_data["samples_count"] == 3

        final_queue_size = message_queue_client.get_queue_size()
        assert final_queue_size == 0

        stats = message_queue_client.get_statistics()
        assert stats["total_produced"] == 3
        assert stats["total_consumed"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
