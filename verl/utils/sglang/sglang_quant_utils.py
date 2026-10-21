# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""SGLang MX low-precision config translation, weight export, and refit registration."""

import os
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from typing import Any

import torch

REFIT_LOADER_FQN = "verl.utils.sglang.sglang_quant_utils.load_and_reprocess"
LOADER_FQN = REFIT_LOADER_FQN
FINALIZE_MARKER = "__verl_sglang_quant_refit_finalize__"
_SYNC_FP8_ATTR = "_verl_sglang_fp8_refit_staged"
_SYNC_FP4_ATTR = "_verl_sglang_fp4_refit_staged"
_SYNC_NVFP4_ATTR = "_verl_sglang_nvfp4_refit_staged"

_REGEX_IGNORE_TO_SGLANG = {
    r"re:.*lm_head$": "lm_head",
    r"re:.*embed_tokens$": "embed_tokens",
    r"re:.*visual\..*": "visual",
    r"re:^mtp\..*": "mtp",
    r"re:.*mlp\.gate$": "mlp.gate",
    r"re:.*shared_expert_gate$": "shared_expert_gate",
    r"re:.*mlp\.experts\..*": "mlp.experts",
    # Qwen3.5 stores these separately in HF/Megatron but SGLang fuses them.
    r"re:.*linear_attn\.in_proj_[ab]$": ("in_proj_ba", "in_proj_b", "in_proj_a"),
}


def resolve_sglang_quantization_config(config: Any, model_config: Any) -> dict[str, Any] | None:
    """Return the rollout override or the model's native quantization config."""
    quantization = getattr(config, "quantization", None)
    if quantization is not None and quantization.enabled:
        return quantization.build_quantization_config(model_config.local_path)

    hf_config = model_config.hf_config
    native = getattr(hf_config, "quantization_config", None)
    if native is None:
        native = getattr(getattr(hf_config, "text_config", None), "quantization_config", None)
    if native is None:
        return None
    if hasattr(native, "to_dict"):
        native = native.to_dict()
    if not isinstance(native, Mapping):
        raise TypeError(f"quantization_config must be a mapping, got {type(native).__name__}")
    return dict(native)


def _translate_ignored_layers(ignored_layers: Iterable[str]) -> list[str]:
    translated = []
    for ignored in ignored_layers:
        if ignored.startswith("re:"):
            native = _REGEX_IGNORE_TO_SGLANG.get(ignored)
            if native is None:
                raise ValueError(
                    f"SGLang native MXFP8 does not support regex ignored layer {ignored!r}; "
                    "use an exact module path or add an engine-name translation"
                )
            native_names = native if isinstance(native, tuple) else (native,)
        else:
            native_names = (ignored,)
        for native_name in native_names:
            if native_name not in translated:
                translated.append(native_name)
    return translated


def translate_mxfp8_to_sglang_fp8_config(ct_config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an all-MXFP8 compressed-tensors override to native SGLang."""
    from verl.utils.qat.compressed_tensors import is_all_mxfp8_config

    if not is_all_mxfp8_config(ct_config):
        raise ValueError("SGLang native MXFP8 translation requires every config group to use MXFP8")
    return {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "ignored_layers": _translate_ignored_layers(ct_config.get("ignore") or []),
    }


class SGLangMxfp8WeightQuantizer:
    """Preserve CT MXFP8 numerics while emitting SGLang's native scale name."""

    def __init__(self, ct_config: Mapping[str, Any]):
        from verl.utils.qat.compressed_tensors import CompressedTensorsWeightQuantizer

        translate_mxfp8_to_sglang_fp8_config(ct_config)
        self.compressed_tensors = CompressedTensorsWeightQuantizer(ct_config)

    def refresh_input_global_scales(self, model: torch.nn.Module) -> None:
        self.compressed_tensors.refresh_input_global_scales(model)

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in self.compressed_tensors(weights):
            if name.endswith(".weight_scale"):
                name = name[: -len("weight_scale")] + "weight_scale_inv"
            yield name, tensor


def check_sglang_mxfp8_support() -> None:
    """Fail before launch when this SGLang build lacks native MXFP8."""
    import sglang
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS

    if "mxfp8" not in QUANTIZATION_METHODS:
        raise RuntimeError(
            f"SGLang {sglang.__version__} does not register native quant_method='mxfp8'; "
            "install a build with MXFP8 Fp8Config support"
        )


def _uses_mxfp4_moe(quantization_config: Mapping[str, Any]) -> bool:
    from verl.utils.qat.compressed_tensors import qat_modes_for_config

    if quantization_config.get("quant_method") != "compressed-tensors":
        return False
    return any(mode is not None and "mxfp4" in mode for mode in qat_modes_for_config(quantization_config))


def _uses_nvfp4_moe(quantization_config: Mapping[str, Any]) -> bool:
    from verl.utils.qat.compressed_tensors import qat_modes_for_config

    if quantization_config.get("quant_method") != "compressed-tensors":
        return False
    return "nvfp4" in qat_modes_for_config(quantization_config)


def uses_sglang_refit_loader(quantization_config: Mapping[str, Any]) -> bool:
    """Whether this config needs verl's bucket-aware SGLang refit loader."""
    from verl.utils.qat.compressed_tensors import is_all_mxfp8_config

    return (
        is_all_mxfp8_config(quantization_config)
        or _uses_mxfp4_moe(quantization_config)
        or _uses_nvfp4_moe(quantization_config)
    )


def _quantization_uses_deep_gemm(quantization_config: Mapping[str, Any]) -> bool:
    """Whether verl supports this SGLang quantization config only through DeepGEMM."""
    values = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                collect(nested)
        elif isinstance(value, str):
            values.append(value.lower())

    collect(quantization_config)
    tokens = set(values)
    if any("nvfp4" in value for value in tokens):
        return False
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method in {"fp8", "mxfp8"}:
        return True
    if any("mxfp4" in value or "mxfp8" in value for value in tokens):
        return True
    if quant_method == "compressed-tensors":
        try:
            from verl.utils.qat.compressed_tensors import qat_modes_for_config

            return bool(qat_modes_for_config(quantization_config) & {"fp8", "mxfp8", "mxfp4", "mxfp4_mxfp8"})
        except (KeyError, TypeError, ValueError):
            return False
    return any(value in {"fp8", "fp8_block_scales"} for value in tokens)


def enforce_sglang_deep_gemm(
    quantization_config: Mapping[str, Any] | None,
    engine_kwargs: dict[str, Any],
) -> None:
    """Pin supported FP8/MXFP8/MXFP4 Linear and MoE kernels to DeepGEMM."""
    if quantization_config is None or not _quantization_uses_deep_gemm(quantization_config):
        return
    for option in ("fp8_gemm_runner_backend", "moe_runner_backend"):
        requested = engine_kwargs.get(option)
        if requested not in (None, "auto", "deep_gemm"):
            raise ValueError(f"SGLang FP8/MXFP8/MXFP4 quantization requires {option}=deep_gemm; got {requested!r}")
        engine_kwargs[option] = "deep_gemm"


def _validate_sglang_nvfp4_cutlass(
    quantization_config: Mapping[str, Any],
    engine_kwargs: dict[str, Any],
) -> None:
    if not _uses_nvfp4_moe(quantization_config):
        return
    requested = engine_kwargs.get("moe_runner_backend")
    if requested not in (None, "auto"):
        raise ValueError(
            "SGLang NVFP4 refit supports only FlashInfer CUTLASS. Leave moe_runner_backend='auto' "
            "so the compressed-tensors NVFP4 scheme selects that runner; "
            f"got {requested!r}"
        )


def prepare_sglang_quantization(
    quantization_config: dict[str, Any],
    custom_weight_loader: list[str],
    engine_kwargs: dict[str, Any],
    *,
    enable_router_replay: bool = False,
) -> dict[str, Any]:
    """Translate supported MX overrides and register their bucket-aware refit loader."""
    from verl.utils.qat.compressed_tensors import is_all_mxfp8_config, qat_modes_for_config

    all_mxfp8 = is_all_mxfp8_config(quantization_config)
    uses_mxfp4 = _uses_mxfp4_moe(quantization_config)
    uses_nvfp4 = _uses_nvfp4_moe(quantization_config)
    enforce_sglang_deep_gemm(quantization_config, engine_kwargs)
    _validate_sglang_nvfp4_cutlass(quantization_config, engine_kwargs)
    if not all_mxfp8 and not uses_mxfp4 and not uses_nvfp4:
        return quantization_config
    if all_mxfp8 and enable_router_replay:
        raise ValueError(
            "SGLang MXFP8 does not currently support rollout routing replay: Qwen3.5-35B-A3B validation "
            "showed that SGLang's captured routes do not match the routes consumed by the native MXFP8 MoE "
            "runner. Disable enable_rollout_routing_replay until that engine path is fixed."
        )
    if REFIT_LOADER_FQN not in custom_weight_loader:
        custom_weight_loader.append(REFIT_LOADER_FQN)

    if all_mxfp8:
        check_sglang_mxfp8_support()
        native_config = translate_mxfp8_to_sglang_fp8_config(quantization_config)
        # ServerArgs spawns fresh scheduler processes. Keep the native config field,
        # and mirror it through SGLang's supported env channel so ignores survive
        # model-config reconstruction in those children.
        existing = [item.strip() for item in os.environ.get("SGLANG_FP8_IGNORED_LAYERS", "").split(",") if item.strip()]
        for ignored in native_config["ignored_layers"]:
            if ignored not in existing:
                existing.append(ignored)
        os.environ["SGLANG_FP8_IGNORED_LAYERS"] = ",".join(existing)
        return native_config

    if uses_nvfp4:
        try:
            from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # noqa: F401
                CompressedTensorsW4A4Nvfp4MoE,
            )
        except ImportError as err:
            raise RuntimeError("This SGLang build does not provide compressed-tensors NVFP4 MoE support") from err
        # SGLang's compressed-tensors resolver raises when an ordinary Linear
        # matches no config group. Ignore every module except the routed-expert
        # prefix, which keeps dense layers in BF16 while allowing the generic
        # NVFP4 MoE scheme (unlike MXFP4, it is selected after ignore checks).
        native_config = deepcopy(quantization_config)
        native_config["ignore"] = list(native_config.get("ignore") or [])
        non_expert = r"re:^(?!.*mlp\.experts(?:\.|$)).*$"
        if non_expert not in native_config["ignore"]:
            native_config["ignore"].append(non_expert)
        return native_config

    modes = qat_modes_for_config(quantization_config)
    if modes == {"mxfp4_mxfp8"}:
        raise NotImplementedError(
            "SGLang MXFP4 refit currently supports BF16 dense layers with MXFP4 routed experts; "
            "the mixed MXFP8-linear + MXFP4-MoE compressed-tensors config is not supported by SGLang 0.5.20"
        )
    try:
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod  # noqa: F401
    except ImportError as err:
        raise RuntimeError("This SGLang build does not provide MXFP4 MoE support") from err

    # SGLang's compressed-tensors resolver raises for a Linear that matches no
    # config group. This override intentionally targets routed experts only;
    # mark every ordinary module name as unquantized. MXFP4 FusedMoE selection
    # runs before the ignore check and is selected from the top-level MXFP4
    # format, so the catch-all does not disable the experts.
    native_config = deepcopy(quantization_config)
    native_config["ignore"] = list(native_config.get("ignore") or [])
    if r"re:.*" not in native_config["ignore"]:
        native_config["ignore"].append(r"re:.*")
    return native_config


def finalize_tensors() -> list[tuple[str, torch.Tensor]]:
    """Return the closing payload sent after the final weight-sync bucket."""
    return [(FINALIZE_MARKER, torch.zeros(1, dtype=torch.uint8))]


def load_and_reprocess(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Load one SGLang quantized-refit bucket and finalize kernel layouts once per sync."""
    from verl.utils.sglang.sglang_fp4_utils import stage_mxfp4_moe_params, stage_nvfp4_moe_params
    from verl.utils.sglang.sglang_fp8_utils import stage_mxfp8_moe_scales

    named_tensors = list(named_tensors)
    finalize = any(name == FINALIZE_MARKER for name, _ in named_tensors)
    named_tensors = [(name, tensor) for name, tensor in named_tensors if name != FINALIZE_MARKER]
    try:
        if named_tensors:
            if _SYNC_FP8_ATTR not in model.__dict__:
                model.__dict__[_SYNC_FP8_ATTR] = stage_mxfp8_moe_scales(model)
                model.__dict__[_SYNC_FP4_ATTR] = stage_mxfp4_moe_params(model)
                model.__dict__[_SYNC_NVFP4_ATTR] = stage_nvfp4_moe_params(model)
            model.load_weights(named_tensors)
        if finalize:
            _finish_sync(model)
    except BaseException:
        _abandon_sync(model)
        raise


def _abandon_sync(model: torch.nn.Module) -> None:
    """Restore CUDA-graph-captured storage after a failed bucket or finalize request."""
    from verl.utils.sglang.sglang_fp4_utils import restore_mxfp4_live, restore_nvfp4_live

    for module, scale_name, live in model.__dict__.pop(_SYNC_FP8_ATTR, ()):
        getattr(module, scale_name).data = live
    restore_mxfp4_live(model.__dict__.pop(_SYNC_FP4_ATTR, ()))
    restore_nvfp4_live(model.__dict__.pop(_SYNC_NVFP4_ATTR, ()))


def _finish_sync(model: torch.nn.Module) -> None:
    """Rebuild FP8/FP4 backend layouts after every bucket has been loaded."""
    from verl.utils.sglang.sglang_fp4_utils import reprocess_mxfp4_moe_layers, reprocess_nvfp4_moe_layers
    from verl.utils.sglang.sglang_fp8_utils import (
        reprocess_mxfp8_layers,
        reprocess_mxfp8_moe_layers,
    )

    if _SYNC_FP8_ATTR not in model.__dict__:
        return
    fp8_staged = model.__dict__[_SYNC_FP8_ATTR]
    fp4_staged = model.__dict__.get(_SYNC_FP4_ATTR, ())
    nvfp4_staged = model.__dict__.get(_SYNC_NVFP4_ATTR, ())
    reprocess_mxfp8_layers(model)
    reprocess_mxfp8_moe_layers(model, fp8_staged)
    reprocess_mxfp4_moe_layers(fp4_staged)
    reprocess_nvfp4_moe_layers(nvfp4_staged)
    del model.__dict__[_SYNC_FP8_ATTR]
    model.__dict__.pop(_SYNC_FP4_ATTR, None)
    model.__dict__.pop(_SYNC_NVFP4_ATTR, None)
