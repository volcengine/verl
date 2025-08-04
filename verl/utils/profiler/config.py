# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass


@dataclass
class NsightToolConfig(BaseConfig):
    """Nsight tool config."""

    "True for each task has its own database, False for all tasks in one training step share one database."
    discrete: bool = False

    def __post_init__(self) -> None:
        pass


@dataclass
class TorchProfilerToolConfig(BaseConfig):
    """Torch profiler tool config.

    Args:
        step_start (int): Start step in update_policy.
        step_end (int): End step.
    """

    step_start: int = -1
    step_end: int = -1

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.step_start, int), f"Profiler step_start must be of type int, got {type(self.step_start)}"
        assert isinstance(self.step_end, int), f"Profiler step_end must be of type int, got {type(self.step_end)}"


@dataclass
class NPUToolConfig(NsightToolConfig):
    """NPU profiler too; config."""

    # options: npu, cpu, memory, shapes, module, stack
    contents: list[str] = field(default_factory=list)

    # Collection level, optional values: level_none, level0, level1, level2.
    level: str = "level1"

    # Whether to automatically parse the data.
    analysis: bool = False

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.contents, list), f"Profiler contents must be of type list, got {type(self.contents)}"
        assert isinstance(self.level, str), f"Profiler level must be of type str, got {type(self.level)}"
        assert isinstance(self.analysis, bool), f"Profiler analysis must be of type bool, got {type(self.analysis)}"
        for content in self.contents:
            assert content in ["npu", "cpu", "memory", "shapes", "module", "stack"], (
                f"Profiler contents only supports npu, cpu, memory, shapes, module, stack, but gets {content}"
            )
        assert self.level in ["level_none", "level0", "level1", "level2"], (
            f"Profiler level only supports level0, 1, 2, and level_none, but gets {self.level}"
        )


@dataclass
class ProfilerConfig(BaseConfig):
    """Worker profiler config.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        discrete (bool): True for each task has its own database, False for all tasks in one training step
          share one database.
        all_ranks (bool): Whether to profile all ranks.
        ranks (list[int]): The ranks that will be profiled. Defaults to [].
    """

    tool: str = None
    enable: bool = False
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    save_path: str = None
    tool_config: NsightToolConfig | TorchProfilerToolConfig | NPUToolConfig = None
    _mutable_fields = ["tool_config"]

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.ranks, set | list | tuple), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )
        if self.tool:
            self.tool_config = omega_conf_to_dataclass(self.tool_config[self.tool])
