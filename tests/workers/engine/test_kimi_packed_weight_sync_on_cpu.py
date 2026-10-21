# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from verl.workers.engine.fsdp.kimi_packed_weights import export_kimi_packed_local_param
from verl.workers.engine.fsdp.utils import unfuse_moe_params


def test_kimi_keeps_offset_aware_packed_expert_transport_abi():
    tensor = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    name = "model.layers.7.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.4"

    updates = list(unfuse_moe_params([(name, tensor)], model_type="kimi_k3"))

    assert len(updates) == 1
    assert updates[0][0] == name
    assert updates[0][1] is tensor


def test_kimi_packed_transport_rejects_other_model_types():
    tensor = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    name = "model.layers.7.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.4"

    with pytest.raises(ValueError, match="valid only for the Kimi K3"):
        list(unfuse_moe_params([(name, tensor)], model_type="legacy"))


def test_kimi_packed_transport_rejects_invalid_offset_before_passthrough():
    tensor = torch.zeros(1, 3, 8)
    name = "model.layers.7.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.bad"

    with pytest.raises(ValueError, match="Invalid Kimi packed-local expert offset"):
        list(unfuse_moe_params([(name, tensor)], model_type="kimi_k3"))


def _check_colocated_export(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2))
        original = torch.arange(4 * 6 * 8).reshape(4, 6, 8).float()
        layout = dict(
            expert_parallel_size=2, tensor_parallel_size=2, data_parallel_size=1, pipeline_model_parallel_size=1
        )
        # Two eFSDP matrix shards x two expert shards. Check initial weights
        # and a changed version, including delivery to both rollout replicas.
        for version in (0, 1):
            weights = original + version
            param = distribute_tensor(weights, mesh, [Shard(1), Shard(0)])
            name, local = export_kimi_packed_local_param("experts.gate_up_proj", param, "cpu", layout)
            start = (rank % 2) * 2
            assert name == f"experts.gate_up_proj.__verl_packed_local__.{start}"
            torch.testing.assert_close(local, weights[start : start + 2], rtol=0, atol=0)
            with patch.object(DTensor, "redistribute", side_effect=AssertionError("must send matrix shards directly")):
                name, local = export_kimi_packed_local_param(
                    "experts.gate_up_proj",
                    param,
                    "cpu",
                    {**layout, "expert_parallel_size": 4, "data_parallel_size": 2},
                )
            assert name.endswith(f".{rank}")
            torch.testing.assert_close(local, weights[rank : rank + 1], rtol=0, atol=0)
        reordered = DeviceMesh("cpu", torch.tensor([[0, 2], [1, 3]]))
        param = distribute_tensor(original, reordered, [Shard(1), Shard(0)])
        _, local = export_kimi_packed_local_param("experts.gate_up_proj", param, "cpu", layout)
        torch.testing.assert_close(local, original[rank % 2 * 2 : rank % 2 * 2 + 2], rtol=0, atol=0)

        # 896 IDs, uneven matrix-row shards, both growing and shrinking EP,
        # source replicas, and buffer chunks that cross expert boundaries.
        original = torch.arange(896 * 7 * 5).reshape(896, 7, 5).float()
        configurations = [
            (torch.arange(4).reshape(2, 2), [Shard(1), Shard(0)], 4),
            (torch.arange(4).reshape(2, 2), [Shard(1), Shard(0)], 1),
            (torch.arange(4), [Shard(0)], 2),
            (torch.arange(4).reshape(2, 2), [Replicate(), Shard(0)], 4),
        ]
        for ranks, placements, target_ep in configurations:
            mesh = DeviceMesh("cpu", ranks)
            for version in (0, 1):
                weights = original + version
                param = distribute_tensor(weights, mesh, placements)
                with (
                    patch("verl.workers.engine.fsdp.kimi_packed_weights._RESHARD_BUFFER_BYTES", 512),
                    patch.object(DTensor, "redistribute", side_effect=AssertionError("must not gather")),
                ):
                    name, local = export_kimi_packed_local_param(
                        "experts.gate_up_proj",
                        param,
                        "cpu",
                        dict(
                            expert_parallel_size=target_ep,
                            tensor_parallel_size=1,
                            data_parallel_size=target_ep,
                            pipeline_model_parallel_size=1,
                        ),
                    )
                count = 896 // target_ep
                start = rank % target_ep * count
                assert name.endswith(f".{start}")
                torch.testing.assert_close(local, weights[start : start + count], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_colocated_export_gathers_matrix_shards_and_keeps_expert_ownership(tmp_path):
    mp.spawn(_check_colocated_export, args=(str(tmp_path / "gloo_rendezvous"),), nprocs=4, join=True)
