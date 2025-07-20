# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import copy
import logging
import os
import random
import tempfile
import time
import unittest

import numpy as np
import torch

from verl import DataProto
from verl.experimental.replay_buffer.persistable_client import PersistableReplayBufferClient
from verl.experimental.replay_buffer.samplers.uniform_key_sampler import UniformKeySampler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)


class TestPersistableReplayBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temp root directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        # Every buffer created in this tests are all under this folder
        base_dir = cls.temp_dir.name
        PersistableReplayBufferClient.MAGIC_SUFFIX = os.path.join(base_dir, PersistableReplayBufferClient.MAGIC_SUFFIX)

    @classmethod
    def tearDownClass(cls):
        # Clean up data in temp_dir after tests finish
        cls.temp_dir.cleanup()

    # Test that pushing and getting are successful and objects are the same before and after
    def test_push_and_get_basic(self):
        # Part 1: Test client.push("rollout_1", batch1); client.get("rollout_1")
        # Same example as verl/.../test_tensor_dict_utilities.py/test_reorder
        client = PersistableReplayBufferClient(replay_buffer_name="test_push", cache_size_limit_in_mb=0.4)
        obs = torch.tensor([1, 2, 3, 4, 5, 6])
        labels = ["a", "b", "c", "d", "e", "f"]
        batch1 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})
        # Batch 1 has ~56 bytes
        client.push("rollout_1", batch1)
        batch1_get = client.get("rollout_1")
        self._validate_correctness([batch1], batch1_get)

        client.push("rollout_1", batch1)  # push twice to the same key
        batch1_get = client.get("rollout_1")
        self._validate_correctness([batch1] * 2, batch1_get)

        # Part 2: Test client.push("rollout_2", [batch1, batch2]); client.get("rollout_2")
        obs = torch.randn(100, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(100)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        batch_list = [batch1, batch2]
        client.push("rollout_2", [batch1, batch2])
        batch_list_get = client.get("rollout_2")
        self._validate_correctness(batch_list, batch_list_get)

        # Part 3: Test client.get("not_existing_rollout")
        res = client.get("not_existing_rollout")
        assert res is None
        client.close()

    # Test that get work correctly when there are pending push tasks
    def test_get_pending_push_tasks(self):
        # Push a large amount of tasks and ensure that p0 queue has unprocessed tasks
        client = PersistableReplayBufferClient(replay_buffer_name="test_get_pending", cache_size_limit_in_mb=0.4)
        list_1 = [12]
        list_2 = [13, 14, 15]
        for _ in range(1000):
            client.push("rollout_1", list_1)
            client.push("rollout_2", list_2)
        # len(p0 queue): 2000, len(p0 index): 2
        expected_result_1 = [12] * 1000
        expected_result_2 = [13, 14, 15] * 1000

        assert client.get("rollout_1") == expected_result_1
        assert client.get("rollout_2") == expected_result_2
        # len(p0 queue): 570, len(p0 index): 2. So get above happened during the processing of p0 queue.
        client.close()

    # Test that get work correctly when there are pending push/delete tasks
    def test_get_pending_tasks(self):
        # Push a large amount of tasks and ensure that p0 queue has unprocessed tasks
        client = PersistableReplayBufferClient(replay_buffer_name="test_get_pendings", cache_size_limit_in_mb=0.4)
        list_1 = [12]
        list_2 = [13, 14, 15]
        for _ in range(500):
            client.delete("rollout_1")
            client.push("rollout_1", list_1)
            client.push("rollout_2", list_2)
            client.delete("rollout_2")
        expected_result_1 = [12]
        expected_result_2 = None

        assert client.get("rollout_1") == expected_result_1
        assert client.get("rollout_2") == expected_result_2
        client.close()

    # Test that uniform sampling works correctly using simple test cases
    def test_uniform_sample_basic(self):
        samplers = [UniformKeySampler()]
        client = PersistableReplayBufferClient(
            replay_buffer_name="test_uniform_sample", cache_size_limit_in_mb=0.4, samplers=samplers
        )
        sample = samplers[0].sample()
        with self.assertRaises(StopIteration):
            next(sample)

        list_1 = [12]
        list_2 = [13, 14, 15]
        client.push("rollout_1", list_1)
        client.push("rollout_2", list_2)
        sample = samplers[0].sample()
        # sampler gets a screenshot of the current keys, so call sample() again after pushing new keys
        # Sample and ensure that every call on next() returns either list_1 or list_2. Uniform sample doesn't
        # raises StopIteration as long as the replay buffer is not empty initially.
        for _ in range(20):
            key = next(sample)
            sampled_list = client.get(key)
            assert sampled_list == list_1 or sampled_list == list_2
        client.close()

    def test_delete_basic(self):
        client = PersistableReplayBufferClient(replay_buffer_name="test_delete", cache_size_limit_in_mb=0.4)
        list_1 = [12]
        client.push("rollout_1", list_1)
        client.delete("rollout_1")
        assert client.get("rollout_1") is None
        client.close()

    # Test that delete successfully delete from both dict and rocksdb when there exists elements on rocksdb
    def test_delete_from_rocksdb(self):
        # Same example as in test_write_eviction. If test_write_eviction passes, then:
        # 'rollout_2' exists in both in-memory cache and rocksdb. 'rollout_1' exists in rocksdb but not in-memory cache.
        client = PersistableReplayBufferClient(replay_buffer_name="test_delete_rocksdb", cache_size_limit_in_mb=0.4)
        obs = torch.randn(1000, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(1000)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        batch_list = [batch2]
        client.push("rollout_1", batch_list)
        client.push("rollout_2", batch_list)

        client.delete("rollout_1")
        client.delete("rollout_2")
        assert client.get("rollout_1") is None and client.get("rollout_2") is None

        self._wait_for_p0_tasks(client)
        rocksdb_value = client._db.get(b"rollout_1")
        dict_value = client._cache._data.get("rollout_1")
        assert rocksdb_value is None and dict_value is None

        rocksdb_value = client._db.get(b"rollout_2")
        dict_value = client._cache._data.get("rollout_2")
        assert rocksdb_value is None and dict_value is None
        client.close()

    # Test that the in-memory dict is lru
    def test_dict_lru(self):
        client = PersistableReplayBufferClient(replay_buffer_name="test_dict_lru", cache_size_limit_in_mb=0.4)
        list_1 = [12]
        list_2 = [13, 14, 15]
        list_3 = [16, 17, 18]
        client.push("rollout_1", list_1)
        client.push("rollout_2", list_2)
        client.push("rollout_3", list_3)
        self._wait_for_p0_tasks(client)
        # Check that the order is {'rollout_1', 'rollout_2', 'rollout_3'} in dict, most recently pushed on the rightmost
        expected_order = ["rollout_1", "rollout_2", "rollout_3"]
        actual_order = list(
            client._cache._data.keys()
        )  # ordereddict.keys() returns a view object that preserves the ordering
        assert expected_order == actual_order

        # order should be correct after get too
        client.get("rollout_3")
        client.get("rollout_1")
        client.get("rollout_2")
        # Check that the order is {'rollout_3', 'rollout_1', 'rollout_2'} in dict
        expected_order = ["rollout_3", "rollout_1", "rollout_2"]
        actual_order = list(client._cache._data.keys())
        assert expected_order == actual_order
        client.close()

    # Test that eviction works correctly when in-memory cache limit is exceeded
    # Also test that writings always pushes to both the dict and rocksdb
    def test_write_eviction(self):
        # Set the bytes limit to ~52000 bytes
        client = PersistableReplayBufferClient(replay_buffer_name="test2", cache_size_limit_in_mb=0.05)
        obs = torch.randn(300, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(300)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        # batch_list has 43200 bytes. Shouldn't evict.
        batch_list = [copy.deepcopy(batch2) for _ in range(3)]
        client.push("rollout_1", batch_list)
        self._wait_for_p0_tasks(client)
        self._wait_for_p1_tasks(client)  # Wait for evict and size calculation to finish
        rocksdb_value = client._db.get(b"rollout_1")
        dict_value = client._cache._data.get("rollout_1")
        assert rocksdb_value is None and dict_value is not None

        # Pushing a new key mapping to another batch2_list. The total bytes in dict >45000. Should evict rollout_1
        # to rocksdb.
        client.push("rollout_2", copy.deepcopy(batch_list))
        self._wait_for_p0_tasks(client)
        self._wait_for_p1_tasks(client)
        rocksdb_value = client._db.get(b"rollout_1")
        dict_value = client._cache._data.get("rollout_1")
        assert rocksdb_value is not None and dict_value is None

        rocksdb_value = client._db.get(b"rollout_2")
        dict_value = client._cache._data.get("rollout_2")
        assert rocksdb_value is None and dict_value is not None
        client.close()

    # Test that eviction during get works correctly
    def test_get_eviction(self):
        # Set the bytes limit to ~52000 bytes
        client = PersistableReplayBufferClient(replay_buffer_name="test3", cache_size_limit_in_mb=0.05)

        # Same example as in test_write_eviction. If test_write_eviction passes, then:
        # 'rollout_2' exists in both in-memory cache and rocksdb. 'rollout_1' exists in rocksdb but not in-memory cache.
        obs = torch.randn(300, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(300)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        # batch_list has 43200 bytes. Shouldn't evict.
        batch_list = [copy.deepcopy(batch2) for _ in range(3)]
        client.push("rollout_1", batch_list)
        client.push("rollout_2", copy.deepcopy(batch_list))
        self._wait_for_p0_tasks(client)
        self._wait_for_p1_tasks(client)

        # Get 'rollout_1' -> 'rollout_2' should be evicted to rocksdb and 'rollout_1' should exist in both in-memory
        # cache and rocksdb.
        client.get("rollout_1")
        self._wait_for_p1_tasks(client)  # Wait for population task to finish
        self._wait_for_eviction_tasks(client)  # wait for eviction manager thread to finish calculating
        self._wait_for_p1_tasks(client)  # Wait for eviction task to finish
        rocksdb_value = client._db.get(b"rollout_1")
        dict_value = client._cache._data.get("rollout_1")
        assert rocksdb_value is None and dict_value is not None

        rocksdb_value = client._db.get(b"rollout_2")
        dict_value = client._cache._data.get("rollout_2")
        assert rocksdb_value is not None and dict_value is None
        client.close()

    # Test that eviction works correctly when there are large number of eviction tasks
    def test_large_scale_eviction(self):
        client = PersistableReplayBufferClient(replay_buffer_name="test_large_eviction", cache_size_limit_in_mb=0.05)
        obs = torch.randn(100, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(100)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        # batch2 bytes:  ~4800
        for i in range(50):
            client.push(f"rollout_{i}", batch2)
        # # wait until eviction manager finishes calculating size
        self._wait_for_p0_tasks(client)
        self._wait_for_p1_tasks(client)
        # check that the lrucache doesn't exceed limit now
        assert not client._cache.size_exceeds_limit()
        client.close()

    # Test that push/get/sample are correct when the key is int
    def test_int_key_type(self):
        samplers = [UniformKeySampler()]
        client = PersistableReplayBufferClient(
            replay_buffer_name="test_int_key", cache_size_limit_in_mb=0.4, samplers=samplers
        )

        list_1 = [1, 2, 3]
        list_2 = [2, 3, 4]
        list_3 = [15]
        client.push(1, list_1)
        client.push(2, list_2)
        client.push(3, list_3)
        assert client.get(1) == list_1 and client.get(2) == list_2 and client.get(3) == list_3

        sample = samplers[0].sample()
        key = next(sample)
        sampled_list = client.get(key)
        assert key in {1, 2, 3}

        for _ in range(2):
            key = next(sample)
            sampled_list = client.get(key)
            assert sampled_list == list_1 or sampled_list == list_2 or sampled_list == list_3
        client.close()

    # Test that sampling works correctly when there is only one key
    def test_sampler_one_key(self):
        samplers = [UniformKeySampler()]
        client = PersistableReplayBufferClient(
            replay_buffer_name="test_one_key", cache_size_limit_in_mb=0, samplers=samplers
        )
        client.push("rollout_1", [1])
        sample_uniform = samplers[0].sample()

        key_uniform = next(sample_uniform)
        assert key_uniform == "rollout_1"
        client.close()

    # Test that reading purely from rocksdb works correctly
    def test_pure_rocksdb(self):
        client = PersistableReplayBufferClient(replay_buffer_name="test4", cache_size_limit_in_mb=0)

        obs = torch.randn(1000, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(1000)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})
        batch_list = [batch2] * 600

        client.push("rollout_1", batch_list)
        self._wait_for_p0_tasks(client)
        self._wait_for_p1_tasks(client)

        batch_list_get = client.get("rollout_1")
        self._validate_correctness(batch_list_get, batch_list)
        client.close()

    # Test that reading partially from rocksdb and partially from cache works correctly
    def test_hybrid(self):
        client = PersistableReplayBufferClient(replay_buffer_name="test5", cache_size_limit_in_mb=0.05)

        obs = torch.randn(1000, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(1000)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})
        batch_list = [batch2] * 600

        # About half in cache, half in rocksdb
        client.push("rollout_1", batch_list)
        client.push("rollout_2", batch_list)
        client.push("rollout_3", batch_list)
        client.push("rollout_4", batch_list)

        # Verify correctness
        batch_list_get = client.get("rollout_1")
        self._validate_correctness(batch_list_get, batch_list)
        batch_list_get = client.get("rollout_2")
        self._validate_correctness(batch_list_get, batch_list)
        client.close()

    # Takes two DataProto lists and validate that they are equal
    def _validate_correctness(self, dp_list: list[DataProto], dp_decoded_list: list[DataProto]):
        assert len(dp_list) == len(dp_decoded_list), "Batch lists length mismatch"
        for i in range(len(dp_list)):
            dp = dp_list[i]
            dp_decoded = dp_decoded_list[i]
            assert dp_decoded.batch.keys() == dp.batch.keys(), "Batch keys mismatch"
            assert dp_decoded.non_tensor_batch.keys() == dp.non_tensor_batch.keys(), "Non-tensor keys mismatch"
            assert dp_decoded.meta_info == dp.meta_info, "Meta info mismatch"
            for key in dp.non_tensor_batch.keys():
                assert np.array_equal(dp.non_tensor_batch[key], dp_decoded.non_tensor_batch[key]), (
                    "Mismatch in non-tensor value"
                )
            for key in dp.batch.keys():
                assert torch.equal(dp.batch[key], dp_decoded.batch[key]), "Mismatch in tensor value"

    # wait for all p0 tasks (push/delete) to finish executing by the task processor
    def _wait_for_p0_tasks(self, client, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            # p0 queue empty means that the task is about to be executed or is executing
            if client._task_processor._p0_q.empty():
                time.sleep(2)  # wait until actual executed
                return
            time.sleep(0.5)
        raise TimeoutError("p0 task queue is not empty after wait")

    # wait for all p1 tasks (evict/populate) to finish executing by the task processor
    def _wait_for_p1_tasks(self, client, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            # p1 queue empty means that the task is about to be executed or is executing
            if client._task_processor._p1_q.empty():
                time.sleep(2)  # wait until actual executed
                return
            time.sleep(0.5)
        raise TimeoutError("p1 task queue is not empty after wait")

    # wait for eviction manager to finish calculating sizes
    def _wait_for_eviction_tasks(self, client, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if client._cache._eviction_manager._work_queue.empty():
                time.sleep(2)  # wait until actual executed
                return
            time.sleep(0.5)
        raise TimeoutError("eviction manager queue is not empty after wait")


if __name__ == "__main__":
    unittest.main(buffer=False)
