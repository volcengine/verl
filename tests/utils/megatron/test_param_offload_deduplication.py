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

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch

from verl.utils import megatron_utils


def _group(is_owner=False):
    return megatron_utils.MegatronParamOffloadGroup(process_group=object(), source_rank=8, is_owner=is_owner)


@pytest.fixture
def cpu_offload(monkeypatch):
    # Exercise storage ownership on CPU without requiring a pinned-memory allocator.
    torch_empty = torch.empty

    def empty_unpinned(*args, **kwargs):
        kwargs["pin_memory"] = False
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_unpinned)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)


@pytest.mark.parametrize("supports_gtp_remat", [False, True])
def test_param_offload_group_builder(monkeypatch, supports_gtp_remat):
    dense_process_group = object()
    expert_process_group = object()
    dense_signature = (
        (lambda with_context_parallel, with_gtp_remat=True: None)
        if supports_gtp_remat
        else (lambda with_context_parallel: None)
    )
    expert_signature = (lambda with_gtp_remat=True: None) if supports_gtp_remat else (lambda: None)
    get_dense_group = create_autospec(dense_signature, return_value=dense_process_group)
    get_expert_group = create_autospec(expert_signature, return_value=expert_process_group)
    monkeypatch.setattr(megatron_utils.parallel_state, "get_data_parallel_group", get_dense_group)
    monkeypatch.setattr(megatron_utils.parallel_state, "get_expert_data_parallel_group", get_expert_group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)

    groups = megatron_utils.build_megatron_param_offload_groups(enabled=True)

    assert groups.dense.process_group is dense_process_group
    assert groups.expert.process_group is expert_process_group
    assert groups.dense.is_owner and groups.expert.is_owner
    assert groups.dense.source_rank == groups.expert.source_rank == 8
    remat_kwargs = {"with_gtp_remat": False} if supports_gtp_remat else {}
    get_dense_group.assert_called_once_with(with_context_parallel=True, **remat_kwargs)
    get_expert_group.assert_called_once_with(**remat_kwargs)


def test_param_offload_group_builder_returns_none_without_replicas(monkeypatch):
    monkeypatch.setattr(megatron_utils, "_get_megatron_param_replica_process_groups", lambda: (object(), object()))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    assert megatron_utils.build_megatron_param_offload_groups(enabled=True) is None


@pytest.mark.parametrize("is_owner", [None, True, False], ids=["default", "owner", "non-owner"])
def test_ddp_offload_round_trip(monkeypatch, cpu_offload, is_owner):
    group = None if is_owner is None else _group(is_owner)
    buffer = SimpleNamespace(param_data=torch.arange(12, dtype=torch.float32))
    broadcast = Mock(side_effect=lambda tensor, **kwargs: tensor.copy_(expected) if is_owner is False else None)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    cpu_copy = None

    for increment in (0, 1):
        buffer.param_data.add_(increment)
        expected = buffer.param_data.clone()
        megatron_utils._offload_ddp_param_buffer(buffer, group)
        megatron_utils._offload_ddp_param_buffer(buffer, group)
        assert buffer.param_data.storage().size() == 0
        if is_owner is False:
            assert buffer.param_data.cpu_data is None
        else:
            if cpu_copy is None:
                cpu_copy = buffer.param_data.cpu_data
            assert buffer.param_data.cpu_data is cpu_copy
            torch.testing.assert_close(cpu_copy, expected)
        megatron_utils._load_ddp_param_buffer(buffer, group)
        torch.testing.assert_close(buffer.param_data, expected)

    assert broadcast.call_count == (0 if group is None else 2)
    if group is not None:
        assert broadcast.call_args.kwargs == {"src": group.source_rank, "group": group.process_group}


def test_deduplicated_buffer_requires_its_group_on_reload():
    buffer = SimpleNamespace(param_data=torch.ones(4))
    megatron_utils._offload_ddp_param_buffer(buffer, _group())

    with pytest.raises(RuntimeError, match="missing parameter offload group"):
        megatron_utils._load_ddp_param_buffer(buffer, None)

    assert buffer.param_data.storage().size() == 0


def test_repeated_default_offload_validates_cpu_backing_size():
    param_data = torch.ones(4)
    param_data.cpu_data = torch.ones(3)
    buffer = SimpleNamespace(param_data=param_data, param_data_size=4)
    param_data.storage().resize_(0)

    with pytest.raises(AssertionError):
        megatron_utils._offload_ddp_param_buffer(buffer, None)


def test_expert_parameter_uses_expert_replica_group():
    dense_group = _group()
    expert_group = _group()
    groups = megatron_utils.MegatronParamOffloadGroups(dense=dense_group, expert=expert_group)
    dense = torch.nn.Parameter(torch.ones(1))
    expert = torch.nn.Parameter(torch.ones(1))
    expert.allreduce = False

    assert megatron_utils._parameter_offload_group(dense, groups) is dense_group
    assert megatron_utils._parameter_offload_group(expert, groups) is expert_group
    assert megatron_utils._parameter_offload_group(dense, None) is None


@pytest.mark.parametrize("is_owner", [None, True, False], ids=["default", "owner", "non-owner"])
@pytest.mark.parametrize("restore_offloaded", [False, True])
def test_ddp_snapshot_round_trip(monkeypatch, cpu_offload, is_owner, restore_offloaded):
    expected = torch.arange(12, dtype=torch.float32)
    buffer = SimpleNamespace(param_data=expected.clone())
    model = SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[])
    monkeypatch.setattr(megatron_utils, "DDP", SimpleNamespace)
    group = None if is_owner is None else _group(is_owner)
    groups = None if group is None else megatron_utils.MegatronParamOffloadGroups(dense=group, expert=None)
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **kwargs: tensor.copy_(expected) if is_owner is False else None
    )

    megatron_utils._offload_ddp_param_buffer(buffer, group)
    state = megatron_utils.copy_megatron_model_to_cpu([model], param_offload_groups=groups)

    saved = state["model_chunk_0"]["buffer_states"][0][0]["param_data"]
    assert (saved is None) == (is_owner is False)
    if groups is not None:
        with pytest.raises(ValueError, match="must match the saved Megatron CPU state"):
            megatron_utils.restore_megatron_model_from_cpu([model], state)
    if not restore_offloaded:
        megatron_utils._load_ddp_param_buffer(buffer, group)
        buffer.param_data.zero_()
    elif saved is not None:
        buffer.param_data.cpu_data.zero_()
    megatron_utils.restore_megatron_model_from_cpu([model], state, param_offload_groups=groups)
    megatron_utils._load_ddp_param_buffer(buffer, group)
    torch.testing.assert_close(buffer.param_data, expected)


@pytest.mark.parametrize("is_owner", [True, False], ids=["owner", "non-owner"])
@pytest.mark.parametrize("restore_offloaded", [False, True])
def test_non_ddp_snapshot_round_trip(monkeypatch, cpu_offload, is_owner, restore_offloaded):
    expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(expected.clone(), requires_grad=False))
    group = _group(is_owner)
    groups = megatron_utils.MegatronParamOffloadGroups(dense=group, expert=None)
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **kwargs: tensor.copy_(expected) if not is_owner else None
    )

    megatron_utils._offload_replicated_parameter(model.weight, group)
    state = megatron_utils.copy_megatron_model_to_cpu([model], param_offload_groups=groups)
    assert (state["model_chunk_0"]["model_state"]["weight"]["data"] is None) == (not is_owner)
    if not restore_offloaded:
        megatron_utils._load_replicated_parameter(model.weight, group, "cpu")
    model.weight.zero_()
    megatron_utils.restore_megatron_model_from_cpu([model], state, param_offload_groups=groups)
    megatron_utils._load_replicated_parameter(model.weight, group, "cpu")
    torch.testing.assert_close(model.weight, expected)
    assert model.weight.stride() == expected.stride()
