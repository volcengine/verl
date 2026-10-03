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

"""Exercise real layout helpers and RouterReplay with CPU Gloo collectives.

Only Megatron import surfaces and parallel-state queries are replaced. This is
not a Megatron model forward or an NCCL/GPU training test.
"""

import importlib.util
import math
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[3] / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_helpers(monkeypatch: pytest.MonkeyPatch, tp_size: int = 1, cp_size: int = 1, cp_rank: int = 0) -> ModuleType:
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.pipeline_parallel.utils",
        "megatron.core.pipeline_parallel.schedules",
        "megatron.core.tensor_parallel",
        "megatron.core.transformer.moe.router",
        "megatron.core.transformer.moe.moe_utils",
        "megatron.core.transformer.moe.token_dispatcher",
        "megatron.core.transformer.transformer_config",
        "megatron.core.transformer.transformer_layer",
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    state = SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: cp_rank,
        get_context_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: tp_size,
    )
    sys.modules["megatron.core"].parallel_state = state
    packed = ModuleType("megatron.core.packed_seq_params")

    class PackedSeqParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    packed.PackedSeqParams = PackedSeqParams
    monkeypatch.setitem(sys.modules, packed.__name__, packed)
    util = _load("verl/models/mcore/util.py", "bshd_mcore_util")
    patch = _load("verl/utils/megatron/router_replay_patch.py", "bshd_router_patch")
    monkeypatch.setitem(sys.modules, "verl.models.mcore.util", util)
    monkeypatch.setitem(sys.modules, "verl.utils.megatron.router_replay_patch", patch)
    helpers = _load("verl/utils/megatron/router_replay_utils.py", "bshd_router_utils")
    helpers.device_name = "cpu"
    helpers.RouterReplayHelper.get_micro_batch_router_list = lambda *_args: patch.RouterReplay.router_instances
    helpers.get_current_rank_layer_info = lambda *_args: {"start": 0, "end": 2}
    helpers.mcore_util = util
    return helpers


def _nested(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _native_topk(scores: torch.Tensor, topk: int, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return scores.topk(topk, dim=1)


def _round_trip(
    helpers: ModuleType,
    tp_size: int,
    cp_size: int,
    tp_rank: int,
    cp_rank: int,
    fp8: str | None,
    forced_max_seqlen: int | None,
    masked: bool,
) -> None:
    lengths = [9, 5, 1]
    ids = _nested([torch.arange(length) + sample * 20 + 1 for sample, length in enumerate(lengths)])
    routes = _nested([torch.stack([row * 8 + 300, row * 8 + 302], dim=-1).unsqueeze(-1) for row in ids.unbind()])
    # Two MoE layers and two experts per token; IDs >255 detect incorrect byte conversion.
    routes = _nested([torch.cat([row, row + 1], dim=-1) for row in routes.unbind()]).to(torch.int16)
    masks = _nested([row % 2 == 0 for row in ids.unbind()])
    config = SimpleNamespace(fp8=fp8, num_layers=2, moe_layer_freq=1)
    alignment = tp_size * cp_size * (2 if cp_size > 1 else 1)
    if fp8 is not None:
        alignment = math.lcm(alignment, 128 * tp_size * cp_size // math.gcd(3, 128 * tp_size * cp_size))
    full_length = math.ceil((forced_max_seqlen or max(lengths)) / alignment) * alignment
    half = full_length // (2 * cp_size)
    positions = (
        list(range(cp_rank * half, (cp_rank + 1) * half))
        + list(range(full_length - (cp_rank + 1) * half, full_length - cp_rank * half))
        if cp_size > 1
        else list(range(full_length))
    )
    # Independent sequence-major oracle: token IDs uniquely identify sample and row.
    token_order = torch.tensor(
        [sample * 20 + pos + 1 if pos < length else 0 for pos in positions for sample, length in enumerate(lengths)]
    )
    token_order = token_order.chunk(tp_size)[tp_rank]
    expected_routes = torch.stack(
        [
            torch.stack([token_order * 8 + 300 + layer * 2, token_order * 8 + 301 + layer * 2], dim=-1)
            for layer in range(2)
        ],
        dim=1,
    )
    valid = token_order != 0
    replay_mask = valid & (token_order % 2 == 0)
    routers = [helpers.RouterReplay() for _ in range(2)]
    for layer, router in enumerate(routers):
        scores = torch.zeros(token_order.numel(), 1024)
        scores.scatter_(1, expected_routes[:, layer], torch.tensor([2.0, 1.0]).expand(token_order.numel(), 2))
        router.set_router_replay_action(helpers.RouterReplayAction.RECORD)
        router.get_replay_topk(scores, 2, default_compute_topk=_native_topk)
    collected = []
    helpers.merge_router_topk_indices(
        None, ids, collected, config, data_format="bshd", forced_max_seqlen=forced_max_seqlen
    )
    assert collected[0].dtype == torch.int16
    for expected, actual in zip(routes.unbind(), collected[0].unbind(), strict=True):
        torch.testing.assert_close(actual, expected)
    helpers.set_router_replay_data(
        collected[0],
        None,
        config,
        replay_mask=masks if masked else None,
        data_format="bshd",
        forced_max_seqlen=forced_max_seqlen,
    )
    for layer, router in enumerate(routers):
        torch.testing.assert_close(router.target_topk_idx[valid], expected_routes[valid, layer])
        if masked:
            torch.testing.assert_close(router.target_replay_mask, replay_mask)
        scores = torch.zeros(token_order.numel(), 1024)
        scores[:, :2] = torch.tensor([2.0, 1.0])
        for action in (helpers.RouterReplayAction.REPLAY_FORWARD, helpers.RouterReplayAction.REPLAY_BACKWARD):
            router.set_router_replay_action(action)
            _, actual = router.get_replay_topk(scores, 2, default_compute_topk=_native_topk)
            selected = replay_mask if masked else valid
            torch.testing.assert_close(actual[selected], expected_routes[selected, layer])
            if masked:
                torch.testing.assert_close(actual[~selected], torch.tensor([0, 1]).expand((~selected).sum(), 2))
        assert not router.replay_backward_list
    helpers.RouterReplay.router_instances.clear()


@pytest.mark.parametrize("fp8", [None, "hybrid"])
@pytest.mark.parametrize("forced_max_seqlen", [None, 17])
@pytest.mark.parametrize("masked", [False, True])
def test_bshd_record_replay_token_identity(
    monkeypatch: pytest.MonkeyPatch, fp8: str | None, forced_max_seqlen: int | None, masked: bool
) -> None:
    helpers = _load_helpers(monkeypatch)
    helpers.gather_from_sequence_parallel_region = lambda tensor, **_kwargs: tensor
    helpers.scatter_to_sequence_parallel_region = lambda tensor: tensor
    _round_trip(helpers, 1, 1, 0, 0, fp8, forced_max_seqlen, masked)


def _thd_round_trip(helpers: ModuleType, dynamic_cp_size: int | None, contiguous: bool) -> None:
    ids = _nested([torch.arange(9), torch.arange(5)])
    routes = _nested([torch.arange(row.numel() * 4).reshape(-1, 2, 2) + 300 for row in ids.unbind()]).to(torch.int16)
    config = SimpleNamespace(fp8=None, num_layers=2, moe_layer_freq=1)
    layout = "contiguous" if contiguous else "zigzag"
    if contiguous:
        config.experimental_attention_variant = "dsv4_hybrid"
        config.csa_window_size = 2
    packed, _, _ = helpers.preprocess_thd_engine(
        routes,
        local_cp_size=dynamic_cp_size,
        cp_layout=layout,
        min_local_rows=2 if contiguous else None,
    )
    local = helpers.scatter_to_sequence_parallel_region(packed.squeeze(0))
    routers = [helpers.RouterReplay() for _ in range(2)]
    for layer, router in enumerate(routers):
        router.record_indices(local[:, layer])
    collected = []
    # Use the existing positional signature; the default must remain THD.
    helpers.merge_router_topk_indices(None, ids, collected, config, None, dynamic_cp_size)
    for expected, actual in zip(routes.unbind(), collected[0].unbind(), strict=True):
        torch.testing.assert_close(actual, expected)
    helpers.set_router_replay_data(collected[0], None, config, None, None, dynamic_cp_size)
    for layer, router in enumerate(routers):
        torch.testing.assert_close(router.target_topk_idx, local[:, layer].long())
    helpers.RouterReplay.router_instances.clear()


def test_legacy_dense_packed_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    helpers = _load_helpers(monkeypatch)
    helpers.gather_from_sequence_parallel_region = lambda tensor, **_kwargs: tensor
    helpers.scatter_to_sequence_parallel_region = lambda tensor: tensor
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
    mask = ids != 0
    routes = torch.arange(32).reshape(2, 4, 2, 2).to(torch.int16) + 300
    packed, _ = helpers.preprocess_packed_seqs(routes, mask)
    routers = [helpers.RouterReplay() for _ in range(2)]
    for layer, router in enumerate(routers):
        router.record_indices(packed[0, :, layer])
    collected = []
    helpers.merge_router_topk_indices(mask, ids, collected, SimpleNamespace(fp8=None))
    torch.testing.assert_close(collected[0][mask], routes[mask])
    helpers.set_router_replay_data(collected[0], mask, SimpleNamespace(fp8=None, num_layers=2, moe_layer_freq=1))
    for layer, router in enumerate(routers):
        torch.testing.assert_close(router.target_topk_idx, packed[0, :, layer].long())


def _distributed_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        for tp_size, cp_size in ((2, 2), (4, 1), (1, 4)):
            tp_groups = [dist.new_group(list(range(cp * tp_size, (cp + 1) * tp_size))) for cp in range(cp_size)]
            cp_groups = [dist.new_group(list(range(tp, 4, tp_size))) for tp in range(tp_size)]
            tp_rank, cp_rank = rank % tp_size, rank // tp_size
            with pytest.MonkeyPatch.context() as monkeypatch:
                helpers = _load_helpers(monkeypatch, tp_size, cp_size, cp_rank)
                helpers.mpu.get_context_parallel_group = lambda group=cp_groups[tp_rank]: group

                def gather(
                    tensor: torch.Tensor, group: Any = tp_groups[cp_rank], size: int = tp_size, **kwargs: Any
                ) -> torch.Tensor:
                    assert tensor.dtype == torch.uint8
                    result = [torch.empty_like(tensor) for _ in range(size)]
                    dist.all_gather(result, tensor, group=group)
                    return torch.cat(result)

                helpers.gather_from_sequence_parallel_region = gather
                helpers.scatter_to_sequence_parallel_region = (
                    lambda tensor, size=tp_size, local_rank=tp_rank: tensor.chunk(size)[local_rank].contiguous()
                )
                helpers.mpu.get_dynamic_data_context_parallel_groups = lambda group_size, group=cp_groups[
                    tp_rank
                ]: group
                for fp8, forced, masked in ((None, None, False), (None, 17, True), ("hybrid", None, True)):
                    _round_trip(helpers, tp_size, cp_size, tp_rank, cp_rank, fp8, forced, masked)
                for dynamic, contiguous in ((None, False), (cp_size, False), (None, True), (cp_size, True)):
                    _thd_round_trip(helpers, dynamic, contiguous)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required for TP/CP layout collectives")
def test_bshd_record_replay_distributed(tmp_path: Path) -> None:
    mp.spawn(_distributed_worker, args=((tmp_path / "gloo_init").as_uri(),), nprocs=4, join=True)
