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

import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.seqlen_balancing import rearrange_micro_batches


class SimpleBatch:
    def __init__(self, mask: torch.Tensor):
        self._mask = mask

    def __getitem__(self, key):
        # support slicing/fancy‐index → return self
        if isinstance(key, (list, torch.Tensor)):
            return self
        if key == "attention_mask":
            return self._mask
        raise KeyError(f"Unexpected key: {key!r}")


def _distributed_worker(rank, world_size, init_method):
    # pin to local GPU
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    # build a per‐rank batch whose total token count grows with rank
    # rank 0 → 1×4 tokens, rank 1 → 3×4 tokens → ceildivs differ
    seq_lens = [1, 3]
    mask = torch.ones((seq_lens[rank], 4), dtype=torch.int, device=f"cuda:{rank}")
    batch = SimpleBatch(mask)
    dp_group = dist.group.WORLD  # just needs to be truthy

    # 1) test min_num_micro_batch
    micros, num_mb, idx = rearrange_micro_batches(
        batch,
        max_token_len=8,  # rank0→ ceildiv(4/8)=1, rank1→ceildiv(12/8)=2
        dp_group=dp_group,
        same_micro_num_in_dp=False,
        min_num_micro_batch=4,  # force at least 4
    )
    assert num_mb == 4
    assert len(micros) == 4
    assert len(idx) == 4

    # 2) test same_micro_num_in_dp
    micros_local, num_local, _ = rearrange_micro_batches(
        batch,
        max_token_len=8,
        dp_group=dp_group,
        same_micro_num_in_dp=False,
        min_num_micro_batch=None,
    )
    expected_local = math.ceil(seq_lens[rank] * 4 / 8)
    assert num_local == expected_local

    micros_global, num_global, _ = rearrange_micro_batches(
        batch,
        max_token_len=8,
        dp_group=dp_group,
        same_micro_num_in_dp=True,
        min_num_micro_batch=None,
    )
    # compute what the *true* global max would be
    per_rank_counts = [math.ceil(seq_len * 4 / 8) for seq_len in seq_lens[:world_size]]
    expected_global = max(per_rank_counts)
    assert num_global == expected_global
    assert len(micros_global) == expected_global

    dist.destroy_process_group()


def test_rearrange_micro_batches_distributed(tmp_path):
    # only spawn 2 procs (you have 8 GPUs available, so this is safe)
    world_size = 2

    # use a file‐based init to synchronize
    init_file = tmp_path / "dist_init"
    init_method = f"file://{init_file}"
    # ensure the file exists
    open(init_file, "a").close()

    mp.spawn(
        _distributed_worker,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True,
    )
