# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Dynamic batching with the BSHD padded-token cost model."""

import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

import verl.utils.device as device_module
import verl.utils.seqlen_balancing as seqlen_balancing
from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import BSHDPadding, rearrange_micro_batches
from verl.workers.engine.utils import prepare_micro_batches


def _make_batch(seq_lens: list[int]) -> TensorDict:
    # The Megatron engine feeds jagged input_ids; each row holds its sample index.
    return tu.get_tensordict(
        {"input_ids": tu.nested_tensor_from_tensor_list([torch.full((n,), i) for i, n in enumerate(seq_lens)])}
    )


def _padded_tokens(partition: list[int], seq_lens: list[int], bshd_padding: BSHDPadding) -> int:
    return bshd_padding.num_tokens(len(partition), max(seq_lens[idx] for idx in partition))


def _check_partitions(
    micro_batches: list[TensorDict],
    partitions: list[list[int]],
    seq_lens: list[int],
    max_token_len: int,
    bshd_padding: BSHDPadding,
) -> None:
    assert sorted(idx for partition in partitions for idx in partition) == list(range(len(seq_lens)))
    for micro_batch, partition in zip(micro_batches, partitions, strict=True):
        assert partition
        assert _padded_tokens(partition, seq_lens, bshd_padding) <= max_token_len
        assert [int(row[0]) for row in micro_batch["input_ids"].unbind()] == partition


def _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size: int, cp_size: int):
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: 0
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    packed_seq_params.PackedSeqParams = type("PackedSeqParams", (), {"__init__": lambda self, **kwargs: None})
    core = types.ModuleType("megatron.core")
    core.parallel_state = parallel_state
    megatron = types.ModuleType("megatron")
    megatron.core = core
    for name, module in {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.parallel_state": parallel_state,
        "megatron.core.packed_seq_params": packed_seq_params,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_bshd_padding", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "use_fp8_padding"),
    [(1, 1, False), (4, 1, False), (1, 8, False), (2, 2, False), (1, 1, True), (2, 2, True)],
)
@pytest.mark.parametrize("seq_lens", [[33, 33], [1, 7, 100], [129, 3, 64, 5, 17]])
@pytest.mark.parametrize("fixed_seq_len", [None, 150])
def test_num_tokens_matches_preprocess_bshd_engine_allocation(
    monkeypatch, tp_size, cp_size, use_fp8_padding, seq_lens, fixed_seq_len
):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=tp_size, cp_size=cp_size)
    input_ids = tu.nested_tensor_from_tensor_list([torch.ones(n, dtype=torch.long) for n in seq_lens])

    input_ids_bshd, _, _ = mcore_util.preprocess_bshd_engine(
        input_ids, use_fp8_padding=use_fp8_padding, forced_max_seqlen=fixed_seq_len
    )

    # Each CP rank holds 1/cp_size of every padded sequence.
    allocated_tokens = input_ids_bshd.shape[0] * input_ids_bshd.shape[1] * cp_size
    bshd_padding = BSHDPadding(
        tp_size=tp_size, cp_size=cp_size, use_fp8_padding=use_fp8_padding, fixed_seq_len=fixed_seq_len
    )
    assert bshd_padding.num_tokens(len(seq_lens), max(seq_lens)) == allocated_tokens


def test_budget_counts_context_parallel_alignment():
    seq_lens = [33, 33]
    # With CP=8 each sequence is padded to a multiple of 16, so the pair stores 2 * 48 = 96 tokens, not 66.
    bshd_padding = BSHDPadding(cp_size=8)

    micro_batches, partitions = rearrange_micro_batches(_make_batch(seq_lens), 95, bshd_padding=bshd_padding)
    assert len(partitions) == 2
    _check_partitions(micro_batches, partitions, seq_lens, 95, bshd_padding)

    _, partitions = rearrange_micro_batches(_make_batch(seq_lens), 96, bshd_padding=bshd_padding)
    assert partitions == [[0, 1]]


def test_mixed_lengths_pack_differently_from_token_model():
    seq_lens = [1024] + [64] * 15
    max_token_len = 2000
    bshd_padding = BSHDPadding()

    _, token_partitions = rearrange_micro_batches(_make_batch(seq_lens), max_token_len)
    micro_batches, partitions = rearrange_micro_batches(_make_batch(seq_lens), max_token_len, bshd_padding=bshd_padding)

    # 1984 real tokens fit one micro-batch, but BSHD pads all 16 samples to 1024 tokens each.
    assert token_partitions == [list(range(16))]
    assert _padded_tokens(token_partitions[0], seq_lens, bshd_padding) == 16 * 1024
    _check_partitions(micro_batches, partitions, seq_lens, max_token_len, bshd_padding)
    assert sorted(partitions) == [[0], list(range(1, 16))]


@pytest.mark.parametrize("bshd_padding", [BSHDPadding(), BSHDPadding(tp_size=2, cp_size=2)])
def test_every_feasible_micro_batch_count_is_valid(bshd_padding):
    rng = random.Random(0)
    seq_lens = [rng.randint(1, 300) for _ in range(24)]
    max_token_len = 1024
    _, partitions = rearrange_micro_batches(_make_batch(seq_lens), max_token_len, bshd_padding=bshd_padding)
    min_count = len(partitions)

    for num_micro_batches in range(min_count, len(seq_lens) + 1):
        micro_batches, partitions = rearrange_micro_batches(
            _make_batch(seq_lens), max_token_len, min_num_micro_batch=num_micro_batches, bshd_padding=bshd_padding
        )
        assert len(partitions) == num_micro_batches
        _check_partitions(micro_batches, partitions, seq_lens, max_token_len, bshd_padding)


def test_fp8_padding_partitions_stay_within_budget():
    rng = random.Random(1)
    seq_lens = [rng.randint(1, 200) for _ in range(16)]
    max_token_len = 1024
    bshd_padding = BSHDPadding(tp_size=2, use_fp8_padding=True)

    for num_micro_batches in range(1, len(seq_lens) + 1):
        micro_batches, partitions = rearrange_micro_batches(
            _make_batch(seq_lens), max_token_len, min_num_micro_batch=num_micro_batches, bshd_padding=bshd_padding
        )
        assert len(partitions) >= num_micro_batches
        _check_partitions(micro_batches, partitions, seq_lens, max_token_len, bshd_padding)


def test_fixed_seq_len_budgets_every_micro_batch_at_that_length():
    seq_lens = [100, 10, 10, 10, 10, 10]
    max_token_len = 250

    _, partitions = rearrange_micro_batches(_make_batch(seq_lens), max_token_len, bshd_padding=BSHDPadding())
    assert len(partitions) == 2

    bshd_padding = BSHDPadding(fixed_seq_len=100)
    micro_batches, partitions = rearrange_micro_batches(_make_batch(seq_lens), max_token_len, bshd_padding=bshd_padding)
    assert len(partitions) == 3
    _check_partitions(micro_batches, partitions, seq_lens, max_token_len, bshd_padding)


def test_single_sample_exceeding_padded_budget_fails_fast():
    # 33 raw tokens fit the budget of 40, but CP=8 pads the sample to 48.
    with pytest.raises(ValueError, match="single sample takes 48 padded BSHD tokens"):
        rearrange_micro_batches(_make_batch([33, 5]), 40, bshd_padding=BSHDPadding(cp_size=8))


def test_rejects_forced_groups():
    with pytest.raises(ValueError, match="force_group_size"):
        rearrange_micro_batches(_make_batch([4, 4]), 16, force_group_size=2, bshd_padding=BSHDPadding())


def test_token_cost_model_is_unchanged():
    seq_lens = [7, 3, 9, 2, 5, 8, 1, 6, 4, 10]

    _, partitions = rearrange_micro_batches(_make_batch(seq_lens), 20)
    assert partitions == [[1, 2, 7], [3, 4, 5, 8], [0, 6, 9]]
    _, partitions = rearrange_micro_batches(_make_batch(seq_lens), 20, min_num_micro_batch=4)
    assert partitions == [[1, 9], [0, 3, 7], [4, 5, 6], [2, 8]]


def test_prepare_micro_batches_budgets_padded_tokens_per_gpu():
    seq_lens = [33, 33, 20, 3]
    data = _make_batch(seq_lens)
    # The Megatron engine scales the per-GPU budget by the CP size, as each CP rank stores 1/cp of a sequence.
    tu.assign_non_tensor(data, use_dynamic_bsz=True, sp_size=8, max_token_len_per_gpu=12)
    bshd_padding = BSHDPadding(cp_size=8)

    micro_batches, partitions = prepare_micro_batches(data, bshd_padding=bshd_padding)

    _check_partitions(micro_batches, partitions, seq_lens, 96, bshd_padding)
    assert len(partitions) == 2


def _dp_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(backend="gloo", init_method=init_method, world_size=world_size, rank=rank)
    seqlen_balancing.get_device_name = lambda: "cpu"
    rng = random.Random(rank)
    # Ranks differ in batch size and length distribution, so they need different micro-batch counts on their own.
    seq_lens = [rng.randint(1, 64) for _ in range(12)] if rank == 0 else [rng.randint(100, 500) for _ in range(9)]
    if rank == 2:
        seq_lens += [1000]
    max_token_len = 1024
    bshd_padding = BSHDPadding(tp_size=2, cp_size=2)

    micro_batches, partitions = rearrange_micro_batches(
        _make_batch(seq_lens), max_token_len, dp_group=dist.group.WORLD, bshd_padding=bshd_padding
    )

    _check_partitions(micro_batches, partitions, seq_lens, max_token_len, bshd_padding)
    _, local_partitions = rearrange_micro_batches(_make_batch(seq_lens), max_token_len, bshd_padding=bshd_padding)
    counts = [None] * world_size
    dist.all_gather_object(counts, (len(local_partitions), len(partitions)))
    assert len({synced for _, synced in counts}) == 1
    assert len({local for local, _ in counts}) > 1
    assert counts[rank][1] == max(local for local, _ in counts)

    # One rank holding an oversized sample must make every rank raise instead of hanging.
    oversized_seq_lens = [33, 5] if rank == 1 else [5, 5]
    with pytest.raises(ValueError, match="single sample takes 48 padded BSHD tokens"):
        rearrange_micro_batches(
            _make_batch(oversized_seq_lens), 40, dp_group=dist.group.WORLD, bshd_padding=BSHDPadding(cp_size=8)
        )

    dist.barrier()
    dist.destroy_process_group()


def test_dp_ranks_agree_on_micro_batch_count_within_padded_budget(tmp_path):
    world_size = 3
    init_file = tmp_path / "dist_init"
    init_file.write_text("")
    mp.spawn(_dp_worker, args=(world_size, f"file://{init_file}"), nprocs=world_size, join=True)
