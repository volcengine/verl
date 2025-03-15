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

from verl.utils.import_utils import (
    is_vllm_available,
    is_sglang_available,
    is_megatron_core_available,
)

from .base import BaseShardingManager
from .fsdp_ulysses import FSDPUlyssesShardingManager

AllGatherPPModel = None

if is_megatron_core_available() and is_vllm_available():
    from .megatron_vllm import AllGatherPPModel, MegatronVLLMShardingManager
elif AllGatherPPModel is not None:
    pass
else:
    AllGatherPPModel = None
    MegatronVLLMShardingManager = None

if is_vllm_available():
    from .fsdp_vllm import FSDPVLLMShardingManager
else:
    FSDPVLLMShardingManager = None

# NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's model_runner would check CUDA device capability.
# However, due to veRL's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
# "RuntimeError: No CUDA GPUs are available".
# For this reason, sharding_manager.__init__ should not import SGLangShardingManager and user need to import use the abs path.
# check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
# if is_sglang_available():
#     from .fsdp.fsdp_sglang import FSDPSGLangShardingManager
# else:
#     FSDPSGLangShardingManager = None
