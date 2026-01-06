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
Compatibility utilities for different versions of transformers library.
"""

import importlib.metadata
import logging
from functools import lru_cache
from typing import Any, Optional

from packaging import version

logger = logging.getLogger(__name__)

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


def resolve_max_model_len_from_hf_config(hf_config: Any) -> int | None:
    mpe = getattr(hf_config, "max_position_embeddings", None)
    if isinstance(mpe, int):
        return mpe
    for subname in ("text_config", "language_config", "llm_config"):
        sub = getattr(hf_config, subname, None)
        mpe = getattr(sub, "max_position_embeddings", None) if sub is not None else None
        if isinstance(mpe, int):
            return mpe
    return None


def maybe_set_max_model_len_from_hf_config(config: Any, hf_config: Any) -> None:
    mpe = resolve_max_model_len_from_hf_config(hf_config)
    if mpe is not None:
        config.max_model_len = mpe
    else:
        logger.warning(
            "Cannot infer max_model_len from hf_config=%s; keeping max_model_len=%s",
            type(hf_config),
            getattr(config, "max_model_len", None),
        )
