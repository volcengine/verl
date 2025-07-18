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
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        save_contents (list[str]): What to include in saved checkpoints.
            Options: 'model', 'optimizer', 'extra', 'hf_model'.
        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
        async_save (bool): Whether to save checkpoints asynchronously.
    """

    _frozen_fields = ["save_contents", "load_contents", "async_save"]

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False


@dataclass
class ProfileConfig(BaseConfig):
    """Configuration for profiling.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        use_profile (bool): Whether to enable profiling.
        profile_ranks (Optional[list[int]]): List of ranks to profile. None means all ranks.
        step_start (int): Starting step for profiling.
        step_end (int): Ending step for profiling.
        save_path (Optional[str]): Path to save profiling results.
    """

    use_profile: bool = False
    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None
