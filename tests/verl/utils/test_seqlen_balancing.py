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
