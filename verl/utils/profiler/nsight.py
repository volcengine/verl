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

"""Nsight launch configuration for controllers, worker groups, and Ray actors."""

from omegaconf import OmegaConf


def _config_select(config, path: str, default=None):
    """Select from DictConfig, mappings, or structured config objects."""
    if config is None:
        return default
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, path, default=default)
    value = config
    missing = object()
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, missing)
        else:
            value = getattr(value, part, missing)
        if value is missing:
            return default
    return value


def _profile_rank_selected(profiler_config, rank: int, *, ranks=None) -> bool:
    """Select a rank; callers must check the overall enable gate separately."""
    if _config_select(profiler_config, "all_ranks", default=False):
        return True
    if ranks is None:
        ranks = _config_select(profiler_config, "ranks", default=[])
    if ranks:
        return rank in ranks
    # Preserve DistProfiler's default: select rank 0 only when profiling is enabled.
    return rank == 0 and _config_select(profiler_config, "enable", default=False)


def _rollout_replica_ranks(ranks, replica_world_size: int) -> list[int]:
    """Map requested global GPU ranks to their owning rollout replicas."""
    return sorted({int(rank) // replica_world_size for rank in ranks})
