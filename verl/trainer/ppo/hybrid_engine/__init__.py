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

from verl.utils.import_utils import is_vllm_available, is_megatron_core_available

from .base import BaseShardingManager

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
