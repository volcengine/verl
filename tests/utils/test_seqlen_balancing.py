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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.utils.model import create_random_mask
from verl.utils.seqlen_balancing import ceildiv, get_reverse_idx, rearrange_micro_batches


def test_seqlen_balancing():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5
    )
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, micro_bsz_idx_lst = rearrange_micro_batches(dataproto.batch, max_token_len=300)
    batch = torch.cat(micro_batches)
    micro_bsz_idx = []
    for idx in micro_bsz_idx_lst:
        micro_bsz_idx.extend(idx)
    reverse_idx_map = get_reverse_idx(micro_bsz_idx)
    reverse_idx_map = torch.tensor(reverse_idx_map)
    new_batch = batch[reverse_idx_map]
    torch.testing.assert_close(new_batch, dataproto.batch)


def _worker(rank, world_size, init_method, max_token_len, use_same_dp, min_mb):
    # 1) init process group & CUDA
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    # 2) build a small random batch (each rank different length to force mismatch)
    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, 10, (20 + rank * 5, 100), device=f"cuda:{rank}")
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_left_padding=0.1,
        max_ratio_of_valid_token=0.9,
        min_ratio_of_valid_token=0.5,
    )
    dp = {"input_ids": input_ids, "attention_mask": attention_mask}
    proto = DataProto.from_single_dict(dp)
    batch = proto.batch

    # 3) call rearrange_micro_batches with one of the two params under test
    micros, idx_lst = rearrange_micro_batches(
        batch,
        max_token_len=max_token_len,
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=use_same_dp,
        min_num_micro_batch=min_mb,
    )

    # 4) check the enforced counts
    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    total_seqlen = seq_len_effective.sum().item()
    local = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))

    if min_mb is not None:
        expected = max(local, min_mb)
        assert len(micros) == expected
    if use_same_dp:
        # gather all local_counts
        counts = [torch.zeros(1, device=f"cuda:{rank}") for _ in range(world_size)]
        counts[rank].fill_(local)
        dist.all_gather(counts, counts[rank])
        expected = max(int(c.item()) for c in counts)
        assert len(micros) == expected
    else:
        # if neither, we get the local natural count
        assert len(micros) == local

    # 5) reconstruction sanity: concat→reverse_idx→orig
    flat = torch.cat(micros, dim=0)
    idx = []
    for sub in idx_lst:
        idx.extend(sub)
    inv = get_reverse_idx(idx)
    inv = torch.tensor(inv, device=flat.device)
    reconstructed = flat[inv]
    torch.testing.assert_close(reconstructed, batch)

    dist.destroy_process_group()


def test_seqlen_balancing_distributed_params(tmp_path):
    world_size = 2
    init_file = tmp_path / "dist_init"
    init_file.write_text("")  # empty file
    init_method = f"file://{init_file}"

    # test min_num_micro_batch only
    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, False, 4),
        nprocs=world_size,
        join=True,
    )

    # test same_micro_num_in_dp only
    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, True, None),
        nprocs=world_size,
        join=True,
    )
