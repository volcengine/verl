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
from verl.utils.profiler import ProfilerConfig

from .model import HFModelConfig
from .rollout import SamplingConfig, ServerConfig

__all__ = ["SandboxFusionConfig", "RewardModelConfig"]


@dataclass
class SandboxFusionConfig(BaseConfig):
    """Configuration for cloud/local sandbox fusion.

    Args:
        url (Optional[str]): Cloud/local function URL for sandbox execution.
        max_concurrent (int): Max concurrent requests allowed to sandbox.
        memory_limit_mb (int): Max memory limit for each sandbox process in MB.
    """

    url: Optional[str] = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    enable: bool = False
    model_type: str = "discriminative"
    name: str = "sglang"
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 2
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    engine_kwargs: dict = field(default_factory=dict)
    max_num_seqs: int = 1024

    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    input_model_config: HFModelConfig = field(default_factory=HFModelConfig)
    model_config: HFModelConfig = field(default_factory=HFModelConfig)
    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)
