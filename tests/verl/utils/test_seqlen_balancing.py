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


import pytest

from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def test_get_seqlen_balanced_partitions_equal_size():
    seqlen_list = [10, 20, 70, 80, 100, 120]
    k_partitions = 3
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=True)

    assert len(partitions) == k_partitions
    all_indices = set()
    partition_sums = []
    for p in partitions:
        assert len(p) == len(seqlen_list) // k_partitions  # Check equal size
        all_indices.update(p)
        partition_sums.append(sum(seqlen_list[i] for i in p))

    assert all_indices == set(range(len(seqlen_list)))  # Check all indices covered
    # Check balance (sums should be close) - allow some tolerance
    assert max(partition_sums) - min(partition_sums) <= max(seqlen_list)  # Heuristic check


def test_get_seqlen_balanced_partitions_unequal_size():
    seqlen_list = [5, 10, 15, 20, 25, 100]
    k_partitions = 2
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=False)

    assert len(partitions) == k_partitions
    all_indices = set()
    partition_sums = []
    for p in partitions:
        assert len(p) > 0  # Check not empty
        all_indices.update(p)
        partition_sums.append(sum(seqlen_list[i] for i in p))

    assert all_indices == set(range(len(seqlen_list)))  # Check all indices covered
    # Check balance (sums should be close)
    assert max(partition_sums) - min(partition_sums) <= max(seqlen_list)  # Heuristic check


def test_get_seqlen_balanced_partitions_assertions():
    with pytest.raises(AssertionError):
        get_seqlen_balanced_partitions([1, 2], 3, False)  # n < k

    with pytest.raises(AssertionError):
        get_seqlen_balanced_partitions([1, 2, 3], 2, True)  # n % k != 0 for equal_size
