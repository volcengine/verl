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

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from verl.utils.profiler.config import NsightToolConfig, ProfilerConfig
from verl.utils.profiler.ray_profile import _RayActorProfilerMixin


@pytest.mark.parametrize(
    "config",
    [
        ProfilerConfig(
            tool="nsys", enable=True, ranks=[2], save_path="traces", tool_config={"nsys": {"discrete": False}}
        ),
        OmegaConf.create(
            {
                "tool": "nsys",
                "enable": True,
                "ranks": [2],
                "save_path": "traces",
                "tool_config": {"nsys": {"discrete": False}},
            }
        ),
    ],
    ids=["dataclass", "omegaconf"],
)
def test_init_profiler_converts_nsight_config(config):
    worker = _RayActorProfilerMixin()
    with patch("verl.utils.profiler.ray_profile.DistProfiler") as profiler:
        worker.init_profiler(2, config, role="reward_loop")

    converted = profiler.call_args.kwargs["config"]
    assert isinstance(converted, ProfilerConfig)
    assert converted.tool == "nsys"
    assert converted.enable is True
    assert converted.ranks == [2]
    assert converted.save_path == "traces"
    tool_config = profiler.call_args.kwargs["tool_config"]
    assert isinstance(tool_config, NsightToolConfig)
    assert tool_config.discrete is False
    profiler.assert_called_once_with(rank=2, config=converted, tool_config=tool_config, save_file_prefix="reward_loop")
    assert worker.profiler is profiler.return_value


def test_init_profiler_defaults_nsight_tool_config():
    worker = _RayActorProfilerMixin()
    config = ProfilerConfig(tool="nsys", tool_config=None)
    with patch("verl.utils.profiler.ray_profile.DistProfiler") as profiler:
        worker.init_profiler(0, config, role="reward_loop")

    profiler.assert_called_once_with(
        rank=0, config=config, tool_config=NsightToolConfig(), save_file_prefix="reward_loop"
    )


def test_init_profiler_ignores_nsight_config_for_other_tools():
    worker = _RayActorProfilerMixin()
    config = ProfilerConfig(tool=None, tool_config={"nsys": {"discrete": True}})
    with patch("verl.utils.profiler.ray_profile.DistProfiler") as profiler:
        worker.init_profiler(0, config, role="reward_loop")

    profiler.assert_called_once_with(rank=0, config=config, tool_config=None, save_file_prefix="reward_loop")


def test_start_profile_forwards_arguments():
    worker = _RayActorProfilerMixin()
    worker.profiler = Mock()

    worker.start_profile(role="reward_loop", profile_step="3")

    worker.profiler.start.assert_called_once_with(role="reward_loop", profile_step="3")


@pytest.mark.parametrize("kwargs, run_command", [({}, True), ({"run_command": False}, False)])
def test_stop_profile_forwards_finish_hook_control(kwargs, run_command):
    worker = _RayActorProfilerMixin()
    worker.profiler = Mock()

    worker.stop_profile(**kwargs)

    worker.profiler.stop.assert_called_once_with(run_command=run_command)
