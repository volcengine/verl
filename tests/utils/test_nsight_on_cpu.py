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

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from verl.utils.profiler.config import ProfilerConfig
from verl.utils.profiler.nsight import (
    _config_select,
    _count_capture_ranges,
    _profile_rank_selected,
    _resolve_nsight_capture_options,
    _rollout_replica_ranks,
    build_controller_nsight_launch_options,
    is_controller_nsys_api_control_enabled,
    resolve_ray_actor_profiler_config,
)


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


@pytest.mark.parametrize(
    "steps, continuous, discrete, expected",
    [
        (None, False, False, 0),
        ([], True, False, 0),
        ([1, 2, 4], False, False, 3),
        ([1, 2, 4], True, False, 2),
        ([4, 2, 1, 2], True, False, 2),
        ([1, 1, 2], False, False, 2),
        ([1, 1, 2], False, True, 6 * 2),
        ([1, 1, 2], True, True, 6 * 2),
        ([1, 2, 4], False, True, 6 * 3),
        ([1, 2, 4], True, True, 6 * 3),
    ],
)
def test_count_capture_ranges(steps, continuous, discrete, expected):
    original_steps = copy.deepcopy(steps)
    assert _count_capture_ranges(steps, continuous, discrete=discrete) == expected
    assert steps == original_steps


@pytest.mark.parametrize(
    "options, capture_count, expected",
    [
        (
            {"capture-range": "cudaProfilerApi"},
            7,
            {"capture-range": "cudaProfilerApi", "capture-range-end": "repeat-shutdown:7"},
        ),
        (
            {"capture-range": "cudaProfilerApi", "capture-range-end": None},
            7,
            {"capture-range": "cudaProfilerApi", "capture-range-end": "repeat-shutdown:7"},
        ),
        (
            {"capture-range": "cudaProfilerApi", "capture-range-end": "stop"},
            None,
            {"capture-range": "cudaProfilerApi", "capture-range-end": "stop"},
        ),
        ({"capture-range": "nvtx"}, None, {"capture-range": "nvtx"}),
        ({}, None, {}),
        ({"capture-range": "cudaProfilerApi"}, 0, {"capture-range": "cudaProfilerApi"}),
    ],
    ids=["missing-end", "null-end", "explicit-end", "nvtx", "empty", "zero-captures"],
)
def test_resolve_nsight_capture_options(options, capture_count, expected):
    with patch("verl.utils.profiler.nsight._count_capture_ranges", return_value=capture_count) as count_ranges:
        result = _resolve_nsight_capture_options(options, [1], False)
    # None marks cases that do not need a capture count; do not constrain those calls.
    if capture_count is not None:
        count_ranges.assert_called_once_with([1], False)
    assert result == expected


@pytest.mark.parametrize(
    "options",
    [
        {"capture-range": "cudaProfilerApi", "o": "worker"},
        OmegaConf.create({"capture-range": "cudaProfilerApi", "o": "worker"}),
    ],
    ids=["dict", "omegaconf"],
)
def test_resolve_nsight_capture_options_returns_independent_dict(options):
    original_options = copy.deepcopy(options)
    with patch("verl.utils.profiler.nsight._count_capture_ranges", return_value=7):
        result = _resolve_nsight_capture_options(options, [1], False)
    assert type(result) is dict
    assert result is not options
    assert result["o"] == "worker"
    result["o"] = "changed"
    assert options == original_options


@pytest.mark.parametrize(
    "profiler_config, expected_options",
    [
        (
            ProfilerConfig(tool="nsys", ray_actor_nsight_options={"o": "role"}),
            {"o": "role"},
        ),
        (
            OmegaConf.create({"tool": "nsys", "ray_actor_nsight_options": {"o": "role"}}),
            {"o": "role"},
        ),
        (
            ProfilerConfig(tool="nsys"),
            {"o": "global"},
        ),
        (ProfilerConfig(tool="nsys", ray_actor_nsight_options={}), {}),
    ],
    ids=["role-options-dataclass", "role-options-omegaconf", "global-fallback", "empty-role-options"],
)
def test_resolve_ray_actor_profiler_config(profiler_config, expected_options):
    config = OmegaConf.create(
        {
            "global_profiler": {
                "steps": [1],
                "profile_continuous_steps": False,
                "global_tool_config": {"nsys": {"worker_nsight_options": {"o": "global"}}},
            }
        }
    )
    original_role = copy.deepcopy(profiler_config)
    original_global = copy.deepcopy(config)
    with patch(
        "verl.utils.profiler.nsight._resolve_nsight_capture_options", return_value={"o": "resolved"}
    ) as resolve_options:
        result = resolve_ray_actor_profiler_config(profiler_config, config)
    resolve_options.assert_called_once_with(expected_options, [1], False)
    assert type(result) is type(profiler_config)
    assert result is not profiler_config
    assert result.ray_actor_nsight_options == {"o": "resolved"}
    result.ray_actor_nsight_options["o"] = "changed"
    assert profiler_config == original_role
    assert config == original_global


@pytest.mark.parametrize("profiler_config", [None, ProfilerConfig(tool="torch"), ProfilerConfig(tool="nsys")])
def test_resolve_ray_actor_profiler_config_without_nsight_options_is_unchanged(profiler_config):
    assert resolve_ray_actor_profiler_config(profiler_config, {}) == profiler_config


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, False),
        ({"tool": "nsys"}, False),
        ({"tool": "nsys", "global_tool_config": {"nsys": {"controller_nsight_options": None}}}, False),
        (
            {"tool": "nsys", "global_tool_config": {"nsys": {"controller_nsight_options": {"capture-range": "nvtx"}}}},
            False,
        ),
        (
            {
                "tool": "torch",
                "global_tool_config": {"nsys": {"controller_nsight_options": {"capture-range": "cudaProfilerApi"}}},
            },
            False,
        ),
        (
            {
                "tool": "nsys",
                "global_tool_config": {"nsys": {"controller_nsight_options": {"capture-range": "cudaProfilerApi"}}},
            },
            True,
        ),
    ],
)
def test_is_controller_nsys_api_control_enabled(settings, expected):
    config = OmegaConf.create({"global_profiler": settings})
    assert is_controller_nsys_api_control_enabled(config) is expected


def test_build_controller_nsight_launch_options():
    config = OmegaConf.create(
        {
            "global_profiler": {
                "steps": [1],
                "profile_continuous_steps": False,
                "global_tool_config": {"nsys": {"controller_nsight_options": {}}},
            }
        }
    )
    with patch(
        "verl.utils.profiler.nsight._resolve_nsight_capture_options", return_value={"o": "resolved"}
    ) as resolve_options:
        result = build_controller_nsight_launch_options(config)
    resolve_options.assert_called_once_with({}, [1], False)
    assert result == {"o": "resolved"}
