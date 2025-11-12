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

import importlib
import os
from functools import cache
from typing import Callable, Optional

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_torch_flex_attn_available,
    logging,
)

_logger = logging.get_logger(__name__)

_ATTN_FUNCS_MAP: Optional[dict[str, Callable]] = None

_HF_VALID_IMPLS = {
    "flash_attention_3",
    "flash_attention_2",
    "flex_attention",
    "sdpa",
    "eager",
}

ENV_ATTN_IMPLEMENTATION = os.getenv("VERL_ATTN_IMPLEMENTATION")

_FA_REQUIRED = (
    "_index_first_axis",
    "lazy_import_flash_attention",
)

_FA_OPTIONAL = (
    "_flash_attention_forward",
    "fa_peft_integration_check",
    "flash_attn_supports_top_left_mask",
)


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"fa", "fa2", "flash2"}:
        return "flash_attention_2"
    if v in {"fa3", "flash3"}:
        return "flash_attention_3"
    if v in {"fi", "flashinfer", "flash_infer"}:
        return "flashinfer"
    if v in {"flex", "flex_attn"}:
        return "flex_attention"
    return v


@cache
def resolve_attn_implementation(preferred: Optional[str] = None, *, for_vision: bool = False) -> str:
    if preferred not in {"auto", *_HF_VALID_IMPLS}:
        _logger.warning_once(f"Unknown attention implementation '{preferred}'. Using auto detection instead.")
        preferred = "auto"
    elif preferred in {*_HF_VALID_IMPLS}:
        if preferred == "flash_attention_3" and not is_flash_attn_3_available():
            _logger.warning_once("Requested flash_attention_3 but it's not available. Using auto detection instead.")
            preferred = "auto"
        elif preferred == "flash_attention_2" and not is_flash_attn_2_available():
            _logger.warning_once("Requested flash_attention_2 but it's not available. Using auto detection instead.")
            preferred = "auto"
        elif preferred == "flex_attention" and not is_torch_flex_attn_available():
            _logger.warning_once("Requested flex_attention but it's not available. Using auto detection instead.")
            preferred = "auto"
        else:
            return preferred
    if preferred == "auto":
        if is_flash_attn_3_available():
            return "flash_attention_3"
        if is_flash_attn_2_available():
            return "flash_attention_2"
        if is_torch_flex_attn_available():
            return "flex_attention"
    return "sdpa"


@cache
def resolve_sglang_attention_backend(preferred: Optional[str] = None) -> str:
    if preferred in ("flash_attention_3", "flash_attention_2", "auto"):
        return "fa3"
    if preferred == "flashinfer":
        return "flashinfer"
    if preferred == "flex_attention":
        return "flex_attention"
    if preferred in ("sdpa", "eager"):
        return "torch_native"
    return "fa3"


@cache
def resolve_vllm_attention_backend(preferred: Optional[str] = None) -> Optional[str]:
    if preferred in ("flash_attention_3", "flash_attention_2"):
        return "FLASH_ATTN"
    if preferred == "flex_attention":
        return "FLEX_ATTENTION"
    if preferred in ("sdpa", "eager"):
        return "TORCH_SDPA"
    if preferred == "flashinfer":
        return "FLASHINFER"
    return None


def configure_attention(
    preferred: Optional[str] = None,
    *,
    sglang_engine_kwargs: Optional[dict] = None,
    set_vllm_env: bool = False,
    for_vision: bool = False,
) -> dict[str, Optional[str]]:
    """
    Configure attention backend across HF, SGLang and vLLM

    - Where to set:
      - Pass `preferred` here, or set env `VERL_ATTN_IMPLEMENTATION`.
      - Prefer per-role config: `actor.attn_implementation`, `rollout.attn_implementation`,
        `ref.attn_implementation`, `critic.attn_implementation`, `reward_model.attn_implementation`
    - Legal values (case-insensitive):
      - 'flash_attention_3', 'flash_attention_2', 'flex_attention', 'sdpa', 'eager', 'auto'.
      - 'flashinfer' is supported for SGLang/vLLM only.
    - Returns dict with keys: 'hf', 'sglang', 'vllm'.
      If requested, also fills `sglang_engine_kwargs['attention_backend']` and sets `VLLM_ATTENTION_BACKEND`
    """
    # preferred (minus auto) > ENV_ATTN_IMPLEMENTATION > auto
    if (not (preferred_norm := _normalize(preferred))) or preferred_norm == "auto":
        preferred_norm = _normalize(ENV_ATTN_IMPLEMENTATION) or "auto"

    hf_impl = (
        preferred_norm
        if ((not _has_assigned_gpu()) and preferred_norm in _HF_VALID_IMPLS and preferred_norm != "auto")
        else resolve_attn_implementation(preferred_norm, for_vision=for_vision)
    )
    sglang_backend = resolve_sglang_attention_backend(preferred_norm)
    vllm_backend = resolve_vllm_attention_backend(preferred_norm)

    if (
        preferred_norm is not None
        and sglang_engine_kwargs is not None
        and "attention_backend" not in sglang_engine_kwargs
    ):
        sglang_engine_kwargs["attention_backend"] = sglang_backend

    if set_vllm_env and vllm_backend is not None and not os.getenv("VLLM_ATTENTION_BACKEND"):
        os.environ["VLLM_ATTENTION_BACKEND"] = vllm_backend

    return {"hf": hf_impl, "sglang": (sglang_backend if preferred_norm is not None else None), "vllm": vllm_backend}


def _get_attention_functions() -> dict[str, Callable]:
    """Resolve attention helpers via Transformers + einops and return a cached mapping.

    - "index_first_axis", "pad_input", "rearrange", "unpad_input":
      backend-agnostic helpers resolved via Transformers (FA3/FA2/NPU or PyTorch fallbacks)
      and einops for tensor rearrangements.
    - `_FA_OPTIONAL` (e.g. "_flash_attention_forward", "fa_peft_integration_check",
      "flash_attn_supports_top_left_mask"):
      HF shims that raise ImportError at call time if unavailable, preserving lazy semantics.

    - required symbols listed in `_FA_REQUIRED` are resolved eagerly and must be present
    - optional symbols listed in `_FA_OPTIONAL` resolve to call-time raising stubs if missing

    The result is cached on first call.
    """
    global _ATTN_FUNCS_MAP

    if _ATTN_FUNCS_MAP is not None:
        return _ATTN_FUNCS_MAP

    try:
        fa_utils = importlib.import_module("transformers.modeling_flash_attention_utils")
    except Exception as e:
        raise ImportError("transformers is required for attention utilities. Please install `transformers`.") from e
    try:
        from einops import rearrange as _einops_rearrange
    except Exception as e:
        raise ImportError("einops is required for tensor rearrangements. Please `pip install einops`.") from e

    def _get_symbol(name: str, *, required: bool) -> Callable:
        sym = getattr(fa_utils, name, None)
        if sym is None:
            if required:
                raise ImportError(f"Required symbol '{name}' not found in transformers.modeling_flash_attention_utils")

            def _raise(*_a, **_k):
                raise ImportError(f"Symbol '{name}' not found in transformers.modeling_flash_attention_utils")

            return _raise
        return sym

    resolved_required = {name: _get_symbol(name, required=True) for name in _FA_REQUIRED}
    resolved_optional = {name: _get_symbol(name, required=False) for name in _FA_OPTIONAL}

    # Select appropriate pad/unpad pair and kwargs processor based on available backends (FA3/FA2/NPU)
    (flash_fn, _flash_varlen_fn, pad_input, unpad_input), process_flash_kwargs_fn = resolved_required[
        "lazy_import_flash_attention"
    ](implementation=None)
    index_first_axis_fn = resolved_required["_index_first_axis"]

    _ATTN_FUNCS_MAP = {
        "index_first_axis": index_first_axis_fn,
        "pad_input": pad_input,
        "rearrange": _einops_rearrange,
        "unpad_input": unpad_input,
        "flash_attn_varlen_func": _flash_varlen_fn,
        "process_flash_kwargs_fn": process_flash_kwargs_fn,
    }
    for _name in _FA_OPTIONAL:
        _ATTN_FUNCS_MAP[_name] = resolved_optional[_name]

    return _ATTN_FUNCS_MAP


def _flash_attention_forward(*args, **kwargs):
    return _get_attention_functions()["_flash_attention_forward"](*args, **kwargs)


def flash_attn_varlen_func(*args, **kwargs):
    return _get_attention_functions()["flash_attn_varlen_func"](*args, **kwargs)


def fa_peft_integration_check(*args, **kwargs):
    return _get_attention_functions()["fa_peft_integration_check"](*args, **kwargs)


def index_first_axis(*args, **kwargs):
    return _get_attention_functions()["index_first_axis"](*args, **kwargs)


def pad_input(*args, **kwargs):
    return _get_attention_functions()["pad_input"](*args, **kwargs)


def rearrange(*args, **kwargs):
    return _get_attention_functions()["rearrange"](*args, **kwargs)


def unpad_input(*args, **kwargs):
    return _get_attention_functions()["unpad_input"](*args, **kwargs)


def flash_attn_supports(feature: str) -> bool:
    feature_norm = (feature or "").strip().lower()
    if feature_norm in {"top_left_mask", "top-left-mask", "top_left"}:
        return _get_attention_functions()["flash_attn_supports_top_left_mask"]()

    process_fn = _get_attention_functions().get("process_flash_kwargs_fn")
    if process_fn is None:
        return False

    try:
        base = dict(query_length=1, key_length=1, is_causal=True, dropout=0.0)
        probe = {}
        if feature_norm in {"window_size", "sliding_window", "sliding-window"}:
            probe["sliding_window"] = 16
        elif feature_norm == "deterministic":
            probe["deterministic"] = True
        elif feature_norm == "softcap":
            probe["softcap"] = 1.0
        else:
            return False

        flash_kwargs = process_fn(**base, **probe)
        if feature_norm in {"window_size", "sliding_window", "sliding-window"}:
            return "window_size" in flash_kwargs
        return feature_norm in flash_kwargs
    except Exception:
        return False


def _has_assigned_gpu() -> bool:
    try:
        import ray

        gpu_ids = ray.get_gpu_ids()
        if isinstance(gpu_ids, list | tuple) and len(gpu_ids) > 0:
            return True
    except Exception:
        pass

    return False


__all__ = [
    "index_first_axis",
    "pad_input",
    "rearrange",
    "unpad_input",
    "_flash_attention_forward",
    "flash_attn_varlen_func",
    "flash_attn_supports",
    "fa_peft_integration_check",
    "resolve_attn_implementation",
    "resolve_sglang_attention_backend",
    "resolve_vllm_attention_backend",
    "configure_attention",
]
