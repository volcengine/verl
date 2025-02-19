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
"""
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

from functools import cache
from typing import List


@cache
def is_megatron_core_available():
    try:
        from megatron.core import parallel_state as mpu
        return True
    except ImportError:
        return False


@cache
def is_vllm_available():
    try:
        import vllm
        return True
    except ImportError:
        return False


def import_external_libs(external_libs=None):
    if external_libs is None:
        return
    if not isinstance(external_libs, List):
        external_libs = [external_libs]
    import importlib
    for external_lib in external_libs:
        importlib.import_module(external_lib)
