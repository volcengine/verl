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

from importlib.metadata import version, PackageNotFoundError


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)

# check if TOOL_USE_VLLM is set in the environment variables
import os
TOOL_USE_VLLM = os.getenv('TOOL_USE_VLLM', 'false').lower() == 'true'

if package_version <= '0.6.3':
    vllm_mode = 'customized'
    from .vllm_rollout import vLLMRollout
else:
    vllm_mode = 'spmd'
    if TOOL_USE_VLLM:
        print("NOTE: Using vLLM rollout with self, please make sure you have vllm >= 0.6.4 installed.")
        from .vllm_rollout_spmd import vLLMRolloutWithSelf as vLLMRollout
    else:
        from .vllm_rollout_spmd import vLLMRollout
