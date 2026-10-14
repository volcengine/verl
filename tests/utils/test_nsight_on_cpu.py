# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest
from omegaconf import OmegaConf

from verl.utils.profiler.config import ProfilerConfig
from verl.utils.profiler.nsight import _config_select, _profile_rank_selected, _rollout_replica_ranks


def _as_namespace(values):
    if isinstance(values, dict):
        return SimpleNamespace(**{key: _as_namespace(value) for key, value in values.items()})
    return values


@pytest.mark.parametrize("make_config", [dict, OmegaConf.create, _as_namespace], ids=["dict", "omegaconf", "object"])
@pytest.mark.parametrize("value", [False, 0, None, [1, 2]])
def test_config_select_preserves_existing_values(make_config, value):
    config = make_config({"profiler": {"ranks": value}})
    result = _config_select(config, "profiler.ranks", default="missing")
    assert result == value
    # OmegaConf wraps lists in ListConfig, so compare their contents without requiring the same type.
    if not isinstance(value, list):
        assert type(result) is type(value)


def test_config_select_returns_default_for_none_config():
    sentinel = object()
    assert _config_select(None, "profiler.ranks", default=sentinel) is sentinel


@pytest.mark.parametrize("make_config", [dict, OmegaConf.create, _as_namespace], ids=["dict", "omegaconf", "object"])
@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, "default"),
        ({"profiler": None}, "default"),
        ({"profiler": {}}, "default"),
        ({"profiler": {"ranks": None}}, None),
    ],
    ids=["missing-intermediate-key", "intermediate-none", "missing-leaf-key", "leaf-none"],
)
def test_config_select_handles_missing_and_none_values(make_config, config, expected):
    config = make_config(config)
    assert _config_select(config, "profiler.ranks", default="default") == expected


def test_config_select_traverses_mixed_dict_and_object_layers():
    config = {"profiler": SimpleNamespace(nsys=SimpleNamespace(enable=True))}
    assert _config_select(config, "profiler.nsys.enable") is True


@pytest.mark.parametrize(
    "config, rank, override, expected",
    [
        (None, 0, None, False),
        (ProfilerConfig(enable=True), 0, None, True),
        (ProfilerConfig(enable=True), 1, None, False),
        (ProfilerConfig(enable=False), 0, None, False),
        (ProfilerConfig(enable=True, ranks=[]), 0, None, True),
        (ProfilerConfig(enable=True, ranks=[1, 2]), 1, None, True),
        (ProfilerConfig(enable=True, ranks=[1, 2]), 0, None, False),
        (ProfilerConfig(enable=False, ranks=[1]), 1, None, True),
        (ProfilerConfig(enable=False, all_ranks=True, ranks=[0]), 3, None, True),
        ({"all_ranks": True}, 1, None, True),
        (OmegaConf.create({"all_ranks": True}), 1, None, True),
        (SimpleNamespace(all_ranks=True), 1, None, True),
        (ProfilerConfig(enable=True, ranks=[2]), 1, [1], True),
    ],
)
def test_profile_rank_selection(config, rank, override, expected):
    # Rank selection is separate from the overall enable gate in DistProfiler.
    assert _profile_rank_selected(config, rank, ranks=override) is expected


@pytest.mark.parametrize(
    "ranks, world_size, expected",
    [
        ([], 1, []),
        ([0 * 3 + 0, 1 * 3 + 1, 2 * 3 + 2], 3, [0, 1, 2]),
        ([1, 0], 1, [0, 1]),
        ([1, 1, 1], 1, [1]),
        (["1"], 1, [1]),
    ],
)
def test_rollout_replica_ranks(ranks, world_size, expected):
    original_ranks = list(ranks)
    assert _rollout_replica_ranks(ranks, world_size) == expected
    assert ranks == original_ranks
