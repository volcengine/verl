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

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

from .model import HFModelConfig
from .rollout import SamplingConfig, ServerConfig

__all__ = ["SandboxFusionConfig", "RewardModelDataProcessorConfig", "RewardModelConfig"]


def get_custome_process_fn(file_path, function_name):
    if not file_path:
        return None

    assert function_name is not None
    module_name = f"custom_reward_module_{function_name}"
    module = sys.modules.get(module_name, None)

    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward process function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward preprocess function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)
    return raw_fn


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
class RewardModelDataProcessorConfig(BaseConfig):
    path: Optional[str] = None
    preprocess_fn_name: Optional[str] = None
    postprocess_fn_name: Optional[str] = None

    def get_process_fn(self):
        preprocess_fn = get_custome_process_fn(
            file_path=self.path,
            function_name=self.preprocess_fn_name,
        )
        postprocess_fn = get_custome_process_fn(
            file_path=self.path,
            function_name=self.postprocess_fn_name,
        )
        return preprocess_fn, postprocess_fn


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

    # for generative reward model
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    data_processor_config: RewardModelDataProcessorConfig = field(default_factory=RewardModelDataProcessorConfig)
    max_new_tokens: int = 4096

    engine_kwargs: dict = field(default_factory=dict)
    max_num_seqs: int = 1024

    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    input_model_config: HFModelConfig = field(default_factory=HFModelConfig)
    model_config: HFModelConfig = field(default_factory=HFModelConfig)
    # Server configuration for sglang server mode
    server_config: ServerConfig = field(default_factory=ServerConfig)
