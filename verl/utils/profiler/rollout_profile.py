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

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf


def rollout_profile_args(config: DictConfig, global_step: int = 1) -> dict[str, Any]:
    """
    Generate profiling parameters for different rollout backends (currently supports sglang,
    with vllm extension interface reserved)

    Args:
        config: Global configuration (Hydra DictConfig), must contain rollout related configurations
        global_step: Current training global step number, used to distinguish profile
                      result directories for different steps

    Returns:
        Dictionary of profiling parameters corresponding to the backend

    Raises:
        NotImplementedError: Unsupported rollout backend
        ValueError: Unsupported profiler tool/missing configuration
    """
    backend = config.rollout.name.lower()
    backend_profile_builders = {
        "sglang": _get_sglang_profile_tags,
    }

    if backend not in backend_profile_builders:
        raise NotImplementedError(
            f"Unsupported rollout backend: {config.rollout.name}, "
            f"currently supported: {list(backend_profile_builders.keys())}"
        )

    return backend_profile_builders[backend](config, global_step)


def _get_sglang_profile_tags(config: DictConfig, global_step: int) -> dict[str, Any]:
    """Generate profiling parameters for sglang backend"""
    tool_to_activities = {
        "torch": ["CPU", "GPU"],
        "torch_memory": ["MEM"],
        "cuda": ["CUDA_PROFILER"],
        "RPD": ["RPD"],
    }
    profiler_tool = config.rollout.profiler.tool
    if profiler_tool not in tool_to_activities:
        raise ValueError(
            f"Unsupported profiler tool for sglang backend: {profiler_tool}, \
               supported tools: {list(tool_to_activities.keys())}"
        )

    # Profiling by stage of Prefill or Decode
    profile_by_stage = OmegaConf.select(config, "rollout.profiler.tool_config.torch.profile_by_stage", default=False)
    # Merge profiles from all ranks into a single trace
    merge_profiles = OmegaConf.select(config, "rollout.profiler.tool_config.torch.merge_profiles", default=False)
    rollout_start_step = OmegaConf.select(config, "rollout.profiler.tool_config.torch.step_start", default=1)
    rollout_end_step = OmegaConf.select(config, "rollout.profiler.tool_config.torch.step_end", default=5)
    rollout_num_steps = rollout_end_step - rollout_start_step

    assert rollout_start_step > 0, f"Rollout start step must be greater than 0 for sglang, but got {rollout_start_step}"
    assert rollout_num_steps > 0, f"Rollout num steps must be greater than 0 for sglang, but got {rollout_num_steps}"

    base_save_path = config.rollout.profiler.save_path
    output_dir = os.path.join(base_save_path, f"rollout_step_{global_step}")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "start_step": rollout_start_step,
        "num_steps": rollout_num_steps,
        "activities": tool_to_activities[profiler_tool],
        "with_stack": True,
        "record_shapes": True,
        "output_dir": output_dir,
        "profile_by_stage": profile_by_stage,
        "merge_profiles": merge_profiles,
    }
