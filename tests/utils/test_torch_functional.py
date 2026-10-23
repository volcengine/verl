# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from verl.utils import torch_functional
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.torch_functional import (
    calculate_sum_pi_squared_from_logits,
    distributed_masked_mean,
    distributed_mean_max_min_std,
    expand_as_nested,
    logprobs_from_logits_v2,
    masked_mean,
)


def _worker_mean(rank: int, world_size: int, rendezvous_file: str):
    # 1) set GPU and init NCCL
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    # each rank holds tensor [rank+1]
    local = torch.tensor([float(rank + 1)], device=f"{get_device_name()}:{rank}")
    mean, gmax, gmin, gstd = distributed_mean_max_min_std(local, True, True, True)

    values = [float(i + 1) for i in range(world_size)]
    exp_mean = sum(values) / len(values)
    exp_max = max(values)
    exp_min = min(values)
    var = sum((x - exp_mean) ** 2 for x in values) / (len(values) - 1)
    exp_std = var**0.5

    # all ranks should see the same result
    assert torch.allclose(mean.cpu(), torch.tensor(exp_mean)), f"mean@{rank}"
    assert torch.allclose(gmax.cpu(), torch.tensor(exp_max)), f"max@{rank}"
    assert torch.allclose(gmin.cpu(), torch.tensor(exp_min)), f"min@{rank}"
    assert torch.allclose(gstd.cpu(), torch.tensor(exp_std)), f"std@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "value,mask,gt",
    [
        ([1.0, 2.0, 3.0, 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 1, 0], float("nan")),
    ],
)
def test_masked_mean(value, mask, gt):
    res = masked_mean(torch.tensor(value), torch.tensor(mask))
    gt = torch.tensor(gt)
    assert torch.allclose(res, gt) or (torch.isnan(res) and torch.isnan(gt))


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_mean_max_min_std(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mean")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mean,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


def _worker_mask(rank: int, world_size: int, rendezvous_file: str):
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # build per‐rank tensor and mask
    local_tensor = torch.tensor([rank * 2 + 1.0, rank * 2 + 2.0], device=f"{get_device_name()}:{rank}")
    if rank == 0:
        mask = torch.tensor([1, 0], device=f"{get_device_name()}:{rank}", dtype=torch.float32)
    else:
        mask = torch.tensor([0, 1], device=f"{get_device_name()}:{rank}", dtype=torch.float32)

    gmean = distributed_masked_mean(local_tensor, mask)

    valid_values = [1.0] + [2 * i + 2.0 for i in range(1, world_size)]
    expected_mean = sum(valid_values) / len(valid_values)
    assert torch.allclose(gmean.cpu(), torch.tensor(expected_mean)), f"masked_mean@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_masked_mean(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mask")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mask,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("shape", [(8, 17), (3, 5, 32), (1, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calculate_sum_pi_squared_from_logits(shape, dtype):
    """Σπ² computed from the logsumexp identity must match the naive softmax-then-square."""
    torch.manual_seed(0)
    logits = torch.randn(*shape, dtype=dtype) * 5.0  # broaden range to expose numerical issues

    actual = calculate_sum_pi_squared_from_logits(logits)
    expected = torch.softmax(logits, dim=-1).pow(2).sum(dim=-1)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    # Σπ² is always in (0, 1] for any non-empty distribution.
    assert torch.all(actual > 0)
    assert torch.all(actual <= 1.0 + 1e-5)


def test_calculate_sum_pi_squared_from_logits_extreme_values():
    """Stable under large-magnitude logits where naive exp would overflow."""
    # fp64 so the comparison isn't dominated by fp32 precision near |z|=1000
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -998.0]], dtype=torch.float64)
    actual = calculate_sum_pi_squared_from_logits(logits)
    expected = torch.softmax(logits, dim=-1).pow(2).sum(dim=-1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_expand_as_nested():
    a = torch.randn(2)
    b = torch.randn(3)
    c = torch.randn(4)
    nested_tensor = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
    tensor = torch.tensor([1, 2, 3])

    output = expand_as_nested(tensor, nested_tensor)

    assert output.values().tolist() == [1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert torch.all(output.offsets() == nested_tensor.offsets()).item()

    # test exceptions
    with pytest.raises(AssertionError):
        expand_as_nested(tensor, tensor)

    other_tensor = torch.tensor([1, 2, 3, 4])

    with pytest.raises(AssertionError):
        expand_as_nested(other_tensor, nested_tensor)

    other_tensor = torch.tensor([[1, 2, 3]])

    with pytest.raises(AssertionError):
        expand_as_nested(other_tensor, nested_tensor)

    with pytest.raises(AssertionError):
        expand_as_nested(tensor, nested_tensor.unsqueeze(-1))


def _per_row_logprobs(logits, labels):
    rows = [F.log_softmax(row, dim=-1)[label] for row, label in zip(logits, labels, strict=True)]
    return torch.stack(rows)


@pytest.mark.parametrize(
    "num_tokens,vocab_size,chunk_elements",
    [(37, 1000, 1 << 28), (37, 1000, 4096), (37, 1000, 1), (5, 151936, 1 << 20)],
    ids=["one-chunk", "uneven-chunks", "one-row-per-chunk", "large-vocab"],
)
def test_logprobs_from_logits_v2_bf16_2d_bitwise_matches_per_row(monkeypatch, num_tokens, vocab_size, chunk_elements):
    monkeypatch.setattr(torch_functional, "_LOGPROB_CHUNK_ELEMENTS", chunk_elements)
    device = get_device_name()
    generator = torch.Generator().manual_seed(0)
    base = (torch.randn(num_tokens, vocab_size, generator=generator) * 4).to(torch.bfloat16).to(device)
    labels = torch.randint(0, vocab_size, (num_tokens,), generator=generator).to(device)
    grad_out = torch.randn(num_tokens, generator=generator).to(torch.bfloat16).to(device)

    logits = base.clone().requires_grad_(True)
    ref_logits = base.clone().requires_grad_(True)
    out = logprobs_from_logits_v2(logits, labels)
    ref = _per_row_logprobs(ref_logits, labels)
    out.backward(grad_out)
    ref.backward(grad_out)

    assert torch.equal(out, ref)
    assert torch.equal(logits.grad, ref_logits.grad)


def test_logprobs_from_logits_v2_bf16_2d_does_not_loop_over_tokens(monkeypatch):
    calls = []
    log_softmax = F.log_softmax

    def counting_log_softmax(*args, **kwargs):
        calls.append(args[0].shape)
        return log_softmax(*args, **kwargs)

    monkeypatch.setattr(torch_functional.F, "log_softmax", counting_log_softmax)
    logits = torch.randn(64, 1000).to(torch.bfloat16)
    labels = torch.randint(0, 1000, (64,))
    assert logprobs_from_logits_v2(logits, labels).shape == (64,)
    assert calls == [torch.Size([64, 1000])]
