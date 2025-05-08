# tests/test_distributed_utils.py
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.torch_functional import distributed_masked_mean, distributed_mean_max_min_std


def _worker_mean(rank: int, world_size: int, rendezvous_file: str):
    # 1) set GPU and init NCCL
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # each rank holds tensor [rank+1]
    local = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
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
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # build per‐rank tensor and mask
    local_tensor = torch.tensor([rank * 2 + 1.0, rank * 2 + 2.0], device=f"cuda:{rank}")
    if rank == 0:
        mask = torch.tensor([1, 0], device=f"cuda:{rank}", dtype=torch.float32)
    else:
        mask = torch.tensor([0, 1], device=f"cuda:{rank}", dtype=torch.float32)

    gmean = distributed_masked_mean(local_tensor, mask)
    assert torch.allclose(gmean.cpu(), torch.tensor(2.5)), f"masked_mean@{rank}"

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
