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
"""SGLang MXFP8 config translation, weight export, and refit registration."""

import os
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

MXFP8_REFIT_LOADER_FQN = "verl.workers.rollout.sglang_rollout.mxfp8_refit_loader.load_and_reprocess"

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


def prepare_sglang_quantization(
    quantization_config: dict[str, Any],
    custom_weight_loader: list[str],
    engine_kwargs: dict[str, Any],
    *,
    enable_router_replay: bool = False,
) -> dict[str, Any]:
    """Translate and wire an MXFP8 override; leave every other CT config intact."""
    from verl.utils.qat.compressed_tensors import is_all_mxfp8_config

    if not is_all_mxfp8_config(quantization_config):
        return quantization_config
    if enable_router_replay:
        raise ValueError(
            "SGLang MXFP8 does not currently support rollout routing replay: Qwen3.5-35B-A3B validation "
            "showed that SGLang's captured routes do not match the routes consumed by the native MXFP8 MoE "
            "runner. Disable enable_rollout_routing_replay until that engine path is fixed."
        )
    check_sglang_mxfp8_support()
    if MXFP8_REFIT_LOADER_FQN not in custom_weight_loader:
        custom_weight_loader.append(MXFP8_REFIT_LOADER_FQN)
    backend = engine_kwargs.get("fp8_gemm_runner_backend", engine_kwargs.get("fp8_gemm_backend"))
    if backend == "flashinfer_trtllm":
        raise ValueError(
            "SGLang MXFP8 refit does not support fp8_gemm_backend=flashinfer_trtllm: "
            "that backend shuffles dense weights in place and cannot be safely reprocessed"
        )
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
