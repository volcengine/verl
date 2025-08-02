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

import logging
import random
import unittest

import numpy as np
import torch

from verl import DataProto
from verl.experimental.replay_buffer.vanilla_client import VanillaReplayBufferClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)


class TestVanillaReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.client = VanillaReplayBufferClient()

    # Test that pushing and getting are successful and objects are the same before and after
    def test_push_and_get_basic(self):
        # Part 1: Test client.push("rollout_1", batch1); client.get("rollout_1")
        # Same example as verl/.../test_tensor_dict_utilities.py/test_reorder
        # start_time = time.time()
        obs = torch.tensor([1, 2, 3, 4, 5, 6])
        labels = ["a", "b", "c", "d", "e", "f"]
        batch1 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})
        # Batch 1 has 56 bytes
        self.client.push("rollout_1", batch1)
        batch1_get = self.client.get("rollout_1")
        self._validate_correctness([batch1], batch1_get)

        # Part 2: Test client.push("rollout_2", [batch1, batch2]); client.get("rollout_2")
        obs = torch.randn(100, 10)
        labels = [random.choice(["abc", "cde"]) for _ in range(100)]
        batch2 = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

        batch_list = [batch1, batch2]
        self.client.push("rollout_2", [batch1, batch2])
        batch_list_get = self.client.get("rollout_2")
        self._validate_correctness(batch_list, batch_list_get)

        # Part 3: Test client.get("not_existing_rollout")
        res = self.client.get("not_existing_rollout")
        # print(f"test_push_and_get_basic takes time: {time.time() - start_time}s")
        assert res is None

    # Test that sampling works correctly using simple test cases
    def test_sample_basic(self):
        sampler = self.client.sample()
        with self.assertRaises(StopIteration):
            next(sampler)

        list_1 = [12]
        list_2 = [13, 14, 15]
        self.client.push("rollout_1", list_1)
        self.client.push("rollout_2", list_2)
        sampler = self.client.sample()
        # Sample 20 times and ensure that every call on next() returns either list_1 or list_2
        for _ in range(20):
            key = next(sampler)
            sampled_list = self.client.get(key)
            assert sampled_list == list_1 or sampled_list == list_2

    def _validate_correctness(self, dp_list: list[DataProto], dp_decoded_list: list[DataProto]):
        assert len(dp_list) == len(dp_decoded_list), "Batch lists length mismatch"
        for i in range(len(dp_list)):
            dp = dp_list[i]
            dp_decoded = dp_decoded_list[i]
            assert dp_decoded.batch.keys() == dp.batch.keys(), "Batch keys mismatch"
            assert dp_decoded.non_tensor_batch.keys() == dp.non_tensor_batch.keys(), "Non-tensor keys mismatch"
            assert dp_decoded.meta_info == dp.meta_info, "Meta info mismatch"
            for key in dp.batch.keys():
                assert torch.equal(dp.batch[key], dp_decoded.batch[key]), "Mismatch in tensor value"
            for key in dp.non_tensor_batch.keys():
                assert np.array_equal(dp.non_tensor_batch[key], dp_decoded.non_tensor_batch[key]), (
                    "Mismatch in non-tensor value"
                )


if __name__ == "__main__":
    unittest.main()
