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

"""Validation for routed-expert, per-token NVFP4 training and rollout."""

import fnmatch
from typing import Any

# Megatron module paths of routed-expert GEMMs (grouped or sequential experts). Dense MLPs
# (``mlp.linear_fc1``) and shared experts (``mlp.shared_experts.linear_fc1``) do not match.
ROUTED_EXPERT_PATTERNS = ("*.mlp.experts.*linear_fc1", "*.mlp.experts.*linear_fc2")


def _hf_get(hf_config: Any, name: str, default: Any = None) -> Any:
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


def _text_config(hf_config: Any) -> Any:
    """The decoder config; multimodal HF configs nest it under ``text_config``."""
    text_config = _hf_get(hf_config, "text_config")
    return hf_config if text_config is None else text_config


def real_nvfp4_num_layers(hf_config: Any) -> int:
    return int(_hf_get(_text_config(hf_config), "num_hidden_layers") or 0)


def is_routed_expert_module(module_path: str) -> bool:
    return any(fnmatch.fnmatch(module_path, pattern) for pattern in ROUTED_EXPERT_PATTERNS)


def real_nvfp4_bf16_layer_indices(num_layers: int, num_layers_at_start: int, num_layers_at_end: int) -> list[int]:
    """Decoder layers kept in BF16 by the first/last carve-out."""

    if num_layers_at_start < 0 or num_layers_at_end < 0:
        raise ValueError(
            f"real_nvfp4 BF16 layer carve-out must be non-negative, got {num_layers_at_start}/{num_layers_at_end}"
        )
    if num_layers_at_start + num_layers_at_end >= num_layers:
        raise ValueError(
            f"real_nvfp4 BF16 layer carve-out {num_layers_at_start}/{num_layers_at_end} "
            f"leaves no quantized layer of {num_layers}"
        )
    return list(range(num_layers_at_start)) + list(range(num_layers - num_layers_at_end, num_layers))


def real_nvfp4_quant_recipe_config(num_layers: int, num_layers_at_start: int, num_layers_at_end: int) -> dict:
    """Megatron-Core per-module recipe: routed-expert GEMMs NVFP4, everything else BF16.

    The result is passed to ``RecipeConfig.from_config_dict``. MCore takes the
    first matching matcher, so the carved-out layers precede the expert
    patterns, and a final catch-all keeps attention, dense MLPs and shared
    experts in BF16: a TE module no matcher selects would inherit the layer's
    FP4 autocast. Layer indices in module paths are global only with PP=1,
    which :func:`validate_real_nvfp4_parallelism` enforces.
    """

    def glob(config: str, pattern: str) -> dict:
        return {"config": config, "type": "glob", "pattern": pattern, "enabled": True}

    matchers = {
        f"layer_{index}_bf16": glob("bf16", f"*.layers.{index}.*")
        for index in real_nvfp4_bf16_layer_indices(num_layers, num_layers_at_start, num_layers_at_end)
    }
    matchers["experts_fc1_nvfp4"] = glob("nvfp4", ROUTED_EXPERT_PATTERNS[0])
    matchers["experts_fc2_nvfp4"] = glob("nvfp4", ROUTED_EXPERT_PATTERNS[1])
    matchers["other_bf16"] = glob("bf16", "*")
    return {
        "configs": {
            "bf16": {"transformer_engine_config_type": "TEQuantizationParams", "training_recipe": {}},
            "nvfp4": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp4_quantization_recipe": "nvfp4"},
            },
        },
        "matchers": matchers,
    }


def validate_real_nvfp4_model_contract(hf_config: Any) -> None:
    """real_nvfp4 trains routed experts in NVFP4, so the model must have them."""

    text_config = _text_config(hf_config)
    num_experts = int(_hf_get(text_config, "num_experts") or _hf_get(text_config, "n_routed_experts") or 0)
    if num_experts <= 0:
        raise ValueError(
            "real_nvfp4 requires a routed-expert MoE model; this config declares no experts "
            "(looked for num_experts / n_routed_experts)"
        )


def validate_real_nvfp4_parallelism(*, pipeline_size: int, virtual_pipeline_size: int | None) -> None:
    """Keep layer-index recipes and rank-local attestation within the PP=1 scope."""

    if pipeline_size != 1 or virtual_pipeline_size not in (None, 1):
        raise ValueError(
            "real_nvfp4 currently requires pipeline_model_parallel_size=1 and no virtual pipeline splitting; "
            "layer precision and local attestation use global layer indices, got pp="
            f"{pipeline_size} vpp={virtual_pipeline_size}"
        )


BACKWARD_OVERRIDES = ("dequantized", "high_precision")


def validate_real_nvfp4_te_recipe(recipe: Any) -> None:
    """Validate the effective Transformer Engine NVFP4 recipe.

    TE reads these fields from ``NVTE_*`` environment variables when the recipe
    is constructed, so this catches a worker started without them. Backward
    GEMMs may use the dequantized forward operands or the high-precision ones
    (``NVTE_BACKWARD_OVERRIDE``); every other field is fixed.
    """

    expected = {
        "row_scaled_activation": True,
        "disable_rht": True,
        "disable_stochastic_rounding": True,
        "disable_2d_quantization": True,
        # The NVFP4_PER_TOKEN rollout uses standard NVFP4, not TE's
        # adaptive 4-over-6 representation. Keep training aligned.
        "nvfp4_4over6": "none",
        "nvfp4_4over6_e4m3_use_256": "all",
        "nvfp4_4over6_err_mode": "MAE",
    }
    mismatches = {
        name: {"expected": wanted, "actual": getattr(recipe, name, None)}
        for name, wanted in expected.items()
        if getattr(recipe, name, None) != wanted
    }
    for group_name in ("fp4_quant_fwd_inp", "fp4_quant_fwd_weight", "fp4_quant_bwd_grad"):
        group = getattr(recipe, group_name, None)
        for field_name in ("random_hadamard_transform", "stochastic_rounding", "fp4_2d_quantization"):
            actual = getattr(group, field_name, None)
            if actual is not False:
                mismatches[f"{group_name}.{field_name}"] = {"expected": False, "actual": actual}
    backward_override = getattr(recipe, "backward_override", None)
    if backward_override not in BACKWARD_OVERRIDES:
        mismatches["backward_override"] = {"expected": BACKWARD_OVERRIDES, "actual": backward_override}
    if mismatches:
        raise RuntimeError(f"real_nvfp4 Transformer Engine recipe drifted: {mismatches}")
