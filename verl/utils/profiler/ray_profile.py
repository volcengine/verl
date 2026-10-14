# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Profiler lifecycle for independent Ray actors."""

from verl.utils.config import omega_conf_to_dataclass

from .config import NsightToolConfig, ProfilerConfig
from .profile import DistProfiler


class _RayActorProfilerMixin:
    """Profiler lifecycle shared by independently launched Ray actors."""

    def init_profiler(self, rank: int, global_profiler_config, role: str) -> None:
        config = omega_conf_to_dataclass(global_profiler_config, dataclass_type=ProfilerConfig)
        tool_config = (
            omega_conf_to_dataclass((config.tool_config or {}).get("nsys"), dataclass_type=NsightToolConfig)
            if config.tool == "nsys"
            else None
        )
        self.profiler = DistProfiler(rank=rank, config=config, tool_config=tool_config, save_file_prefix=role)

    def start_profile(self, **kwargs) -> None:
        self.profiler.start(**kwargs)

    def stop_profile(self, run_command: bool = True) -> None:
        self.profiler.stop(run_command=run_command)
