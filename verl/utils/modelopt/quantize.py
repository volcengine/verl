# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""ModelOpt quantization config and application for Megatron QAT."""

import copy

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    is_registered_quant_backend,
    register_quant_backend,
)

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}

_FP8_BLOCK_BACKEND = "verl_fp8_block"
_MXFP4_BACKEND = "verl_mxfp4"
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


class _FP8BlockFakeQuant(torch.autograd.Function):
    """Dynamic FP8 E4M3 fake quantization over ``(rows, cols)`` blocks of the last two dims, straight-through grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_rows: int, block_cols: int) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).float()
        rows, cols = x2d.shape
        pad_rows, pad_cols = -rows % block_rows, -cols % block_cols
        xp = F.pad(x2d, (0, pad_cols, 0, pad_rows))
        blocks = xp.reshape(xp.shape[0] // block_rows, block_rows, xp.shape[1] // block_cols, block_cols)
        scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / _FP8_MAX
        dq = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float() * scale
        dq = dq.reshape(xp.shape)[:rows, :cols]
        return dq.reshape(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def _fp8_block_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    block_rows, block_cols = quantizer.backend_extra_args["block_sizes"]
    return _FP8BlockFakeQuant.apply(inputs, block_rows, block_cols)


class _Mxfp4FakeQuant(torch.autograd.Function):
    """compressed-tensors-compatible MXFP4 fake quantization with STE."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        from compressed_tensors.quantization import QuantizationArgs
        from compressed_tensors.quantization.lifecycle.forward import fake_quantize
        from compressed_tensors.quantization.utils import calculate_qparams

        args = QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            group_size=32,
            strategy="group",
            dynamic=False,
            scale_dtype="torch.uint8",
        )
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        groups = x2d.unflatten(-1, (-1, 32))
        min_vals, max_vals = groups.amin(dim=-1), groups.amax(dim=-1)
        scale, zero_point = calculate_qparams(min_vals.float(), max_vals.float(), args)
        return fake_quantize(x2d, scale, zero_point, args).reshape(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _mxfp4_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    return _Mxfp4FakeQuant.apply(inputs)


if not is_registered_quant_backend(_FP8_BLOCK_BACKEND):
    register_quant_backend(_FP8_BLOCK_BACKEND, _fp8_block_backend)
if not is_registered_quant_backend(_MXFP4_BACKEND):
    register_quant_backend(_MXFP4_BACKEND, _mxfp4_backend)

# Mirrors the compressed-tensors "fp8" rollout scheme: 128x128 weight blocks and
# dynamic 1x128 activation groups, both with fp32 scales. Only linear GEMMs the
# rollout quantizes may be enabled; anything else (attention BMMs, GDN conv1d,
# vision tower) would make training see a different model than the rollout.
_FP8_BLOCK_QUANTIZER_CFG = {
    "*": {"enable": False},
    "*weight_quantizer": {
        "num_bits": (4, 3),
        "backend": _FP8_BLOCK_BACKEND,
        "backend_extra_args": {"block_sizes": [128, 128]},
        "enable": True,
    },
    "*input_quantizer": {
        "num_bits": (4, 3),
        "backend": _FP8_BLOCK_BACKEND,
        "backend_extra_args": {"block_sizes": [1, 128]},
        "enable": True,
    },
    "*conv1d*": {"enable": False},
    "*vision_model*": {"enable": False},
}

_QUANTIZER_CFGS = {
    "w4a16": _NVFP4_W4A16_QUANTIZER_CFG,
    "fp8": _FP8_BLOCK_QUANTIZER_CFG,
}
_SUPPORTED_QAT_MODES = {*_QUANTIZER_CFGS, "mxfp8", "mxfp4", "mxfp4_mxfp8"}

_ROUTED_EXPERT_WEIGHT_QUANTIZER = "*mlp.experts*weight_quantizer"
_ROUTED_EXPERT_INPUT_QUANTIZER = "*mlp.experts*input_quantizer"


def _modelopt_quantizer_cfg(preset: dict, quantizer_name: str) -> dict:
    """Copy one quantizer's numeric config from a native ModelOpt preset."""
    entries = mtq.normalize_quant_cfg_list(copy.deepcopy(preset["quant_cfg"]))
    return copy.deepcopy(next(entry["cfg"] for entry in entries if entry["quantizer_name"] == quantizer_name))


def _mxfp4_quantize_config(*, dense_mxfp8: bool) -> tuple[list[dict], None]:
    """MXFP4 routed-expert weights with MXFP8 activations.

    ModelOpt has native MXFP4/MXFP4 and MXFP8/MXFP8 presets but no MXFP4/MXFP8
    preset. Their quantizers are independent, so combine the weight numerics
    from the former with the input numerics from the latter.
    """
    if dense_mxfp8:
        quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(mtq.MXFP8_DEFAULT_CFG["quant_cfg"]))
    else:
        quant_cfg = [{"quantizer_name": "*", "enable": False}]

    weight_cfg = _modelopt_quantizer_cfg(mtq.MXFP4_DEFAULT_CFG, "*weight_quantizer")
    # ModelOpt and compressed-tensors use different E2M1 tie handling for a
    # small subset of values. Export is compressed-tensors, so QAT must use the
    # same QDQ implementation to remain numerically identical.
    weight_cfg["backend"] = _MXFP4_BACKEND
    quant_cfg.extend(
        [
            {
                "quantizer_name": _ROUTED_EXPERT_WEIGHT_QUANTIZER,
                "cfg": weight_cfg,
            },
            {
                "quantizer_name": _ROUTED_EXPERT_INPUT_QUANTIZER,
                "cfg": _modelopt_quantizer_cfg(mtq.MXFP8_DEFAULT_CFG, "*input_quantizer"),
            },
        ]
    )
    return quant_cfg, None


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg.append({"quantizer_name": key, "enable": False})
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode not in _SUPPORTED_QAT_MODES:
        raise ValueError(f"Supported QAT modes are {sorted(_SUPPORTED_QAT_MODES)}, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    if qat_mode == "mxfp8":
        # ModelOpt's native preset is exactly the compressed-tensors MXFP8
        # numerics: dynamic groups of 32, E4M3 values and E8M0 scales.
        modelopt_config = copy.deepcopy(mtq.MXFP8_DEFAULT_CFG)
        quant_cfg = mtq.normalize_quant_cfg_list(modelopt_config["quant_cfg"])
        algorithm = modelopt_config["algorithm"]
    elif qat_mode in ("mxfp4", "mxfp4_mxfp8"):
        quant_cfg, algorithm = _mxfp4_quantize_config(dense_mxfp8=qat_mode == "mxfp4_mxfp8")
    else:
        quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(_QUANTIZER_CFGS[qat_mode]))
        disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
        if isinstance(disabled_cfg, dict):
            disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
        quant_cfg.extend(disabled_cfg)
        algorithm = "max" if qat_mode == "w4a16" else None
    quant_cfg.extend(ignore_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
