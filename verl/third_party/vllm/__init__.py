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

import logging
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from packaging import version as vs

from verl.utils.device import is_npu_available
from verl.utils.import_utils import is_sglang_available


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


logger = logging.getLogger(__name__)


@lru_cache
def parse_vllm_version(raw: str) -> vs.Version:
    """Parse a vLLM version, treating untagged source builds as the newest API.

    A vLLM built from a checkout without release tags reports setuptools-scm's
    fallback ``0.1.devN`` (optionally with a local ``+...`` suffix), which would
    otherwise compare as older than every release. The string does not tell how old
    the checkout is, so this is an assumption: an untagged build of an old vLLM
    passes the gates that use this parser (minimum versions, the profiler arguments,
    the routing-replay minimum) and can fail later, inside vLLM. It is logged once.
    Gates where the old path is the safe one keep the raw version.
    """
    parsed = vs.parse(raw)
    if parsed.is_devrelease and parsed.release == (0, 1) and parsed.pre is None and parsed.post is None:
        logger.warning(
            "vLLM %s is an untagged source build; verl assumes it has the newest vLLM API, "
            "so minimum-version checks pass it: build from a tagged checkout to have them apply.",
            raw,
        )
        return vs.parse("9999")
    return parsed


package_name = "vllm"
package_version = get_version(package_name)
vllm_version = None
VLLM_SLEEP_LEVEL = 1

if package_version is None:
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.18.0+"
        )
elif is_npu_available:
    # sleep_mode=2 is not supported on vllm-ascend for now, will remove this restriction when this ability is ready.
    VLLM_SLEEP_LEVEL = 1
    from vllm import LLM
    from vllm.distributed import parallel_state
elif parse_vllm_version(package_version) >= vs.parse("0.18.0"):
    vllm_version = package_version
    VLLM_SLEEP_LEVEL = 2
    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    # An unusable vLLM is tolerated when SGLang can serve as the backend instead,
    # matching the branch above for a missing vLLM.
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.18.0+"
        )

__all__ = ["LLM", "parallel_state"]
