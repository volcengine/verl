import copy
import logging
import random
import tempfile
import time

import torch
from pympler import asizeof

from verl import DataProto
from verl.utils.replay_buffer.persistable_replay_buffer_client import PersistableReplayBufferClient
from verl.utils.replay_buffer.samplers.ordered_by_key_sampler import OrderedByKeySampler
from verl.utils.replay_buffer.samplers.uniform_key_sampler import UniformKeySampler
from verl.utils.replay_buffer.task_processor import Task, TaskType
from verl.utils.replay_buffer.vanilla_replay_buffer_client import VanillaReplayBufferClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)

NUM_REQUESTS = 30000
"""
Results (NUM_REQUESTS = 30000, Pushing ~1200MB in total):
After implementing processing engine.

pure_cache:
1. PUSH QPS: 400433
2. GET QPS: 701306
3. SAMPLE QPS: 452287

pure_rocksdb:
1. PUSH QPS: 524318
2. GET QPS: 481101
3. SAMPLE QPS: 304567

hybrid:
1. PUSH QPS: 89824
2. GET QPS: 710529
3. SAMPLE QPS: 477878

vanilla:
PUSH QPS: 924120
GET QPS: 2604995
SAMPLE QPS: 1343180
"""


# Test the performance (qps) of push/get/sample under large dataset situations. Using cache-rocksdb hybrid by default.
class TestIntegrationPerformance:
    def __init__(self, pure_cache=False, pure_rocksdb=False, hybrid=True):  # Only one should be true
        # self.temp_dir = tempfile.TemporaryDirectory()
        # PersistableReplayBufferClient.DB_BASE_DIR = self.temp_dir.name
        if pure_cache:
            cache_size_limit_in_mb = 1500
        elif pure_rocksdb:
            cache_size_limit_in_mb = 0
        else:  # hybrid
            cache_size_limit_in_mb = 600
        self.samplers = [UniformKeySampler()]
        self.client = PersistableReplayBufferClient(
            replay_buffer_name="e2e_test2", cache_size_limit_in_mb=cache_size_limit_in_mb, samplers=self.samplers
        )
        for i in range(NUM_REQUESTS):
            self.client.delete(f"rollout_{i}")

    # Test the qps for push.
    def test_push_qps(self):
        obs = torch.randn(300, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(300)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        # batch_list has 43200 bytes. If NUM_REQUESTS = 30000, then ~1235MB will be pushed in total.
        batch_list = [copy.deepcopy(batch2) for _ in range(3)]

        start_time = time.time()
        for i in range(NUM_REQUESTS):
            self.client.push(f"rollout_{i}", batch_list)
        end_time = time.time()
        print(f"PUSH QPS: {NUM_REQUESTS / (end_time - start_time)}s")

    # Test the qps for get.
    def test_get_qps(self):
        start_time = time.time()
        for i in range(NUM_REQUESTS):
            self.client.get(f"rollout_{i}")
        end_time = time.time()
        print(f"GET QPS: {NUM_REQUESTS / (end_time - start_time)}s")

    # Test the qps for sample.
    def test_sample_qps(self):
        start_time = time.time()
        sample = self.samplers[0].sample()
        for _ in range(NUM_REQUESTS):
            key = next(sample)
            sampled_list = self.client.get(key)
        end_time = time.time()
        print(f"SAMPLE QPS: {NUM_REQUESTS / (end_time - start_time)}s")


if __name__ == "__main__":
    tester = TestIntegrationPerformance(hybrid=True)
    tester.test_push_qps()
    tester.test_get_qps()
    tester.test_sample_qps()
    tester.client.close()
    # Cleans up data
    # tester.temp_dir.cleanup()
    time.sleep(10)
