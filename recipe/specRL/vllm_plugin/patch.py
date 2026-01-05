# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Snowflake Inc.
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
#

import logging
import os

import vllm

logger = logging.getLogger(__name__)


def specRL_plugin():
    """vLLM plugin for FlexFlow.

    This plugin enables FlexFlow to be used with vLLM. It consists of a
    collection of patches that are applied to vLLM at runtime.
    """

    # To enable the plugin, set the environment variable VLLM_PLUGINS=specRL_plugin.
    #
    # The plugin is activated when vLLM is imported. It is only activated in the
    # main process. It is not activated in vLLM's worker processes.

    # The plugin is compatible with vLLM versions 0.3.2 and later.
    # It is not compatible with vLLM versions prior to 0.3.2.

    if os.getenv("VLLM_USE_V1") == "0":
        logger.warning(
            "specRL only supports vLLM V1, but detected V0 engine. "
            "Ignoring plugin!\n"
            "Hint: To strictly enforce the V1 vLLM engine, please set "
            "VLLM_USE_V1=1."
        )
        return

    if vllm.__version__.startswith("0.10.0"):
        from .v0_10_0 import patch
    # elif vllm.__version__.startswith("0.8.3"):
    #     from .v0_8_3 import patch
    else:
        logger.warning(f"specRL requires vllm==0.10.0 but found vllm=={vllm.__version__}. Ignoring plugin!")
        return

    # Patches that make later patches work properly.
    patch.WorkerBasePatch.apply_patch()
