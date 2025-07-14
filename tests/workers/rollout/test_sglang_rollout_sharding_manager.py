import pytest
import torch

from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

_TENSOR_1MB = torch.zeros(512, 512)
_BYTES_1MB = 1 << 20


@pytest.mark.parametrize(
    "named_tensors, bucket_size_mb, gt_groups",
    [
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            0.5 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            1 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            1.5 * _BYTES_1MB,
            [["a"], ["b"]],
        ),
        (
            [("a", _TENSOR_1MB), ("b", _TENSOR_1MB)],
            2 * _BYTES_1MB,
            [["a", "b"]],
        ),
    ],
)
def test_get_named_tensor_buckets(named_tensors, bucket_size_mb, gt_groups: list[list[str]]):
    named_tensors_iter = iter(named_tensors)
    groups = list(get_named_tensor_buckets(named_tensors_iter, bucket_size_mb))
    assert len(groups) == len(gt_groups)
    for group, gt_group in zip(groups, gt_groups, strict=True):
        assert len(group) == len(gt_group)
        for (name, _), (gt_name) in zip(group, gt_group, strict=True):
            assert name == gt_name
