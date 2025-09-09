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

__all__ = ["ServerConfig", "SandboxFusionConfig", "RewardModelConfig"]


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


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
    """Configuration for reward model scoring.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable reward model.
        enable_resource_pool (bool): Whether to deploy the model to a separate resource pool.
        n_gpus_per_node (int): Number of GPUs per node when using resource pool.
        nnodes (int): Number of nodes when using resource pool.
        strategy (str): FSDP strategy: "fsdp" or "fsdp2".
        model (Dict[str, Any]): Model configuration for reward scoring.
        micro_batch_size (Optional[int]): Global micro batch size (deprecated).
        micro_batch_size_per_gpu (Optional[int]): Local per-GPU micro batch size.
        max_length (Optional[int]): Maximum sequence length to process for scoring.
        use_dynamic_bsz (bool): Whether to dynamically adjust batch size at runtime.
        forward_max_token_len_per_gpu (int): Maximum number of tokens per GPU in one forward pass.
        reward_manager (str): Reward manager type (naive or prime).
        launch_reward_fn_async (bool): Whether to launch custom reward function asynchronously during log_prob.
        sandbox_fusion (Dict[str, Any]): Cloud/local sandbox fusion configuration for custom reward logic.
        profiler (Dict[str, Any]): Profiler configuration for reward model.
    """

    _mutable_fields = BaseConfig._mutable_fields

    enable: bool = False
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    # strategy: str = MISSING
    # model: BaseModelConfig = field(default_factory=BaseModelConfig)
    # micro_batch_size: Optional[int] = None
    # micro_batch_size_per_gpu: Optional[int] = None
    # max_length: Optional[int] = None
    # use_dynamic_bsz: bool = False
    # forward_max_token_len_per_gpu: int = 32768
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False

    tensor_model_parallel_size: int = 2
    engine_kwargs: dict = field(default_factory=dict)
    max_num_seqs: int = 1024
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    free_cache_engine: bool = True

    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    input_model_config: HFModelConfig = field(default_factory=HFModelConfig)
    model_config: HFModelConfig = field(default_factory=HFModelConfig)
    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)
