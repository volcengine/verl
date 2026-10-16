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
"""Low-precision rollout expressed as a compressed-tensors ``quantization_config``.

The rollout config either selects a compressed-tensors preset scheme per layer kind::

    quantization:
      linear: FP8_BLOCK  # dense linear layers (attention, dense MLP, shared experts)
      moe: FP8_BLOCK     # routed experts
      ignore: []         # extra "re:" patterns kept in high precision

or provides a complete ``quantization_config`` JSON via ``quantization.config_file``.

``RolloutQuantizationConfig.build_quantization_config`` (via
``build_compressed_tensors_config``) turns that into the HF ``quantization_config``
the inference engine is started with (``load_format=dummy``), and
``CompressedTensorsWeightQuantizer`` quantizes the trainer's exported HF weights
with exactly the same targets and ignore list, using the compressed-tensors
compressors, so every refit ships tensors in the layout the engine loads.
"""

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch

# compressed-tensors ships with the inference engines rather than verl, so it is
# imported lazily to keep this module importable without them.
if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

__all__ = [
    "MOE_TARGET",
    "SUPPORTED_SCHEMES",
    "CompressedTensorsWeightQuantizer",
    "build_compressed_tensors_config",
    "qat_modes_for_config",
]

# Matched against vLLM's unfused expert names (``...mlp.experts.0.gate_proj``),
# which is how compressed-tensors resolves fused MoE layers.
MOE_TARGET = r"re:.*mlp\.experts\..*"

# Presets whose rollout refit path (weight reload + process_weights_after_loading)
# has been validated, mapped to the ModelOpt QAT mode that fake-quantizes identically.
SUPPORTED_SCHEMES: dict[str, Optional[str]] = {
    "FP8_BLOCK": "fp8",
    "MXFP8": "mxfp8",
}

# Layers that stay in high precision regardless of model. Patterns must start
# with "re:": vLLM rewrites plain dotted names through the model's
# WeightsMapper, but leaves regexes alone, so these match on both sides.
DEFAULT_IGNORE = [
    r"re:.*lm_head$",
    r"re:.*embed_tokens$",
    r"re:.*visual\..*",
    r"re:^mtp\..*",
    r"re:.*mlp\.gate$",
    r"re:.*shared_expert_gate$",
]


_HIGH_PRECISION_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _check_scheme(kind: str, scheme: Optional[str]) -> Optional[str]:
    if scheme is None:
        return None
    scheme = scheme.upper()
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported rollout quantization {kind}={scheme!r}, choose from {sorted(SUPPORTED_SCHEMES)}")
    return scheme


_NUMERIC_FIELDS = {
    "num_bits",
    "type",
    "symmetric",
    "group_size",
    "strategy",
    "block_structure",
    "dynamic",
    "scale_dtype",
}


def _numerics(args: Optional["QuantizationArgs"]) -> Optional[dict[str, Any]]:
    return None if args is None else args.model_dump(include=_NUMERIC_FIELDS)


def _preset_of(scheme: "QuantizationScheme") -> Optional[str]:
    """The supported preset ``scheme`` quantizes like, ignoring observers and targets."""
    from compressed_tensors.quantization import preset_name_to_scheme

    for preset in SUPPORTED_SCHEMES:
        reference = preset_name_to_scheme(preset, scheme.targets)
        if all(
            _numerics(getattr(scheme, kind)) == _numerics(getattr(reference, kind))
            for kind in ("weights", "input_activations", "output_activations")
        ):
            return preset
    return None


def _group_schemes(ct_config: Mapping[str, Any]) -> list["QuantizationScheme"]:
    from compressed_tensors.quantization import QuantizationScheme

    return [
        QuantizationScheme.model_validate({k: v for k, v in group.items() if k != "format"})
        for group in ct_config["config_groups"].values()
    ]


def qat_modes_for_config(ct_config: Mapping[str, Any]) -> set[Optional[str]]:
    """ModelOpt QAT modes whose fake quant matches the config groups of a rollout ``quantization_config``."""
    return {SUPPORTED_SCHEMES[_preset_of(scheme)] for scheme in _group_schemes(ct_config)}


def _matches(name: str, targets: Iterable[str]) -> bool:
    for target in targets:
        if target.startswith("re:"):
            if re.match(target[3:], name):
                return True
        elif name == target:
            return True
    return False


def _config_group(targets: list[str], preset: str) -> dict[str, Any]:
    from compressed_tensors.compressors.format import infer_module_format
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme(preset, targets)
    group = scheme.model_dump(mode="json", exclude_none=True)
    group["format"] = infer_module_format(torch.nn.Linear, scheme).value
    return group


def _load_config_file(config_file: str) -> dict[str, Any]:
    """Read a compressed-tensors ``quantization_config`` (bare, or inside an HF ``config.json``)."""
    with open(config_file) as f:
        config = json.load(f)
    config = config.get("quantization_config", config)
    if config.get("quant_method") != "compressed-tensors" or not config.get("config_groups"):
        raise ValueError(f"{config_file} is not a compressed-tensors quantization_config with config_groups")
    for name, scheme in zip(config["config_groups"], _group_schemes(config), strict=True):
        if _preset_of(scheme) is None:
            raise ValueError(
                f"{config_file}: config group {name!r} does not match any supported scheme {sorted(SUPPORTED_SCHEMES)}"
            )
    return config


def build_compressed_tensors_config(
    *,
    linear: Optional[str],
    moe: Optional[str],
    ignore: Iterable[str],
    config_file: Optional[str],
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """Build the HF compressed-tensors ``quantization_config`` for the rollout engine.

    Prefer ``RolloutQuantizationConfig.build_quantization_config``; the arguments mirror its fields.
    """
    ignore = list(ignore or [])
    if config_file:
        if linear or moe or ignore:
            raise ValueError("rollout.quantization.config_file is exclusive with linear / moe / ignore")
        return _load_config_file(config_file)

    linear = _check_scheme("linear", linear)
    moe = _check_scheme("moe", moe)
    if linear is None and moe is None:
        raise ValueError("rollout quantization needs at least one of `linear` / `moe`")

    ignore = DEFAULT_IGNORE + ignore

    config_groups: dict[str, Any] = {}
    if linear == moe:
        config_groups["group_0"] = _config_group(["Linear"], linear)
    else:
        # A regex target wins over the class name, so experts always get the MoE scheme.
        if moe is not None:
            config_groups["group_moe"] = _config_group([MOE_TARGET], moe)
        else:
            ignore.append(MOE_TARGET)
        if linear is not None:
            config_groups["group_linear"] = _config_group(["Linear"], linear)

    formats = {g["format"] for g in config_groups.values()}
    return {
        "quant_method": "compressed-tensors",
        "format": formats.pop() if len(formats) == 1 else "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": config_groups,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "sparsity_config": {},
        "global_compression_ratio": None,
    }


def _weight_min_max(weight: torch.Tensor, args: "QuantizationArgs") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile min/max of a 2D weight, shaped like the scale the strategy expects."""
    from compressed_tensors.quantization import QuantizationStrategy
    from compressed_tensors.quantization.utils import maybe_pad_tensor_for_block_quant

    if args.strategy == QuantizationStrategy.BLOCK:
        rows, cols = args.block_structure
        padded = maybe_pad_tensor_for_block_quant(weight, (rows, cols))
        tiles = padded.unflatten(1, (-1, cols)).unflatten(0, (-1, rows))
        return tiles.amin(dim=(1, 3)), tiles.amax(dim=(1, 3))
    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        groups = weight.unflatten(-1, (-1, args.group_size))
        return groups.amin(dim=-1), groups.amax(dim=-1)
    if args.strategy == QuantizationStrategy.CHANNEL:
        return weight.amin(dim=-1, keepdim=True), weight.amax(dim=-1, keepdim=True)
    if args.strategy == QuantizationStrategy.TENSOR:
        return weight.amin(), weight.amax()
    raise NotImplementedError(f"Trainer-side quantization not implemented for strategy {args.strategy}")


def _compress_weight(weight: torch.Tensor, scheme: "QuantizationScheme") -> dict[str, torch.Tensor]:
    """Quantize one module's weight into its compressed-tensors state dict."""
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.compressors.format import infer_module_format
    from compressed_tensors.quantization import QuantizationStrategy
    from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

    args = scheme.weights
    min_vals, max_vals = (t.float() for t in _weight_min_max(weight, args))
    state_dict = {"weight": weight}
    global_scale = None
    if args.strategy == QuantizationStrategy.TENSOR_GROUP:
        global_scale = generate_gparam(min_vals.amin(), max_vals.amax())
        state_dict["weight_global_scale"] = global_scale
    scale, zero_point = calculate_qparams(min_vals, max_vals, args, global_scale=global_scale)
    state_dict["weight_scale"] = scale
    if not args.symmetric:
        state_dict["weight_zero_point"] = zero_point
    compressor = BaseCompressor.get_value_from_registry(infer_module_format(torch.nn.Linear, scheme).value)
    return compressor.compress(state_dict, scheme)


# HF checkpoints that pack all routed experts into one 3D tensor (e.g. Qwen3.5 MoE).
_PACKED_EXPERTS = re.compile(r"(?P<prefix>.*\.experts)\.(?P<proj>gate_up_proj|down_proj)(?:\.weight)?$")


def _unpack_experts(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """``experts.gate_up_proj`` [E, 2I, H] / ``experts.down_proj`` [E, H, I] -> per-expert 2D ``*.weight``.

    The engines load per-expert names with their scales, while their packed
    3D loaders only map the weight itself.
    """
    match = _PACKED_EXPERTS.match(name)
    prefix, proj = match["prefix"], match["proj"]
    for expert_id, weight in enumerate(tensor):
        if proj == "gate_up_proj":
            gate, up = weight.chunk(2, dim=0)
            yield f"{prefix}.{expert_id}.gate_proj.weight", gate
            yield f"{prefix}.{expert_id}.up_proj.weight", up
        else:
            yield f"{prefix}.{expert_id}.down_proj.weight", weight


class CompressedTensorsWeightQuantizer:
    """Quantize exported HF weights to match a compressed-tensors rollout config.

    A weight is quantized iff it is a 2D ``*.weight`` whose module is not
    ignored and is targeted by a config group, resolved like compressed-tensors
    does: a name / "re:" target wins over the ``Linear`` class target. Packed 3D
    experts are split per expert first. Everything else passes through unchanged.
    """

    def __init__(self, ct_config: Mapping[str, Any]):
        self.ignore = list(ct_config.get("ignore") or [])
        self.schemes = _group_schemes(ct_config)
        self.linear_scheme = next((s for s in self.schemes if "Linear" in s.targets), None)

    def _module_scheme(self, module: str) -> Optional["QuantizationScheme"]:
        if _matches(module, self.ignore):
            return None
        for scheme in self.schemes:
            if _matches(module, scheme.targets):
                return scheme
        return self.linear_scheme

    def _scheme_for(self, name: str, tensor: torch.Tensor) -> Optional["QuantizationScheme"]:
        if not name.endswith(".weight") or tensor.ndim != 2 or tensor.dtype not in _HIGH_PRECISION_DTYPES:
            return None
        return self._module_scheme(name[: -len(".weight")])

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            packed = _PACKED_EXPERTS.match(name) if tensor.ndim == 3 else None
            if packed and self._module_scheme(f"{packed['prefix']}.0.gate_proj") is not None:
                yield from self(_unpack_experts(name, tensor))
                continue
            scheme = self._scheme_for(name, tensor)
            if scheme is None:
                yield name, tensor
                continue
            prefix = name[: -len("weight")]
            for key, value in _compress_weight(tensor, scheme).items():
                yield prefix + key, value
