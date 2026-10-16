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
_MXFP8_BACKEND = "verl_mxfp8"
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


class _MXFP8FakeQuant(torch.autograd.Function):
    """Dynamic MXFP8 fake quantization over groups on the last dimension, with E8M0 scales."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        if x.shape[-1] % group_size:
            raise ValueError(
                f"MXFP8 requires the last dimension ({x.shape[-1]}) to be divisible by group_size={group_size}; "
                "add this layer to the QAT and rollout ignore lists"
            )

        groups = x.float().unflatten(-1, (-1, group_size))
        amax = groups.abs().amax(dim=-1, keepdim=True)

        # compressed-tensors rounds each group amax to E8M0 before applying the
        # FP8 E4M3 element-format offset (2**8). Its power-of-two rounding moves
        # to the next exponent at a 1.75 significand.
        base_exp = torch.floor(torch.log2(amax))
        base = torch.exp2(base_exp)
        rounded_exp = base_exp + (amax >= 1.75 * base)
        encoded_exp = (127 + rounded_exp - 8).clamp(0, 255)
        scale = torch.exp2(encoded_exp - 127)

        dq = (groups / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float() * scale
        return dq.flatten(start_dim=-2).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _mxfp8_backend(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    return _MXFP8FakeQuant.apply(inputs, quantizer.backend_extra_args["group_size"])


if not is_registered_quant_backend(_FP8_BLOCK_BACKEND):
    register_quant_backend(_FP8_BLOCK_BACKEND, _fp8_block_backend)
if not is_registered_quant_backend(_MXFP8_BACKEND):
    register_quant_backend(_MXFP8_BACKEND, _mxfp8_backend)

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

# Mirrors compressed-tensors MXFP8: FP8 E4M3 values in groups of 32 with
# dynamic E8M0 scales for both weights and activations.
_MXFP8_QUANTIZER_CFG = {
    "*": {"enable": False},
    "*weight_quantizer": {
        "num_bits": (4, 3),
        "backend": _MXFP8_BACKEND,
        "backend_extra_args": {"group_size": 32},
        "enable": True,
    },
    "*input_quantizer": {
        "num_bits": (4, 3),
        "backend": _MXFP8_BACKEND,
        "backend_extra_args": {"group_size": 32},
        "enable": True,
    },
    "*conv1d*": {"enable": False},
    "*vision_model*": {"enable": False},
}

_QUANTIZER_CFGS = {
    "w4a16": _NVFP4_W4A16_QUANTIZER_CFG,
    "fp8": _FP8_BLOCK_QUANTIZER_CFG,
    "mxfp8": _MXFP8_QUANTIZER_CFG,
}


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
    if qat_mode not in _QUANTIZER_CFGS:
        raise ValueError(f"Supported QAT modes are {sorted(_QUANTIZER_CFGS)}, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = mtq.normalize_quant_cfg_list(copy.deepcopy(_QUANTIZER_CFGS[qat_mode]))
    disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
    if isinstance(disabled_cfg, dict):
        disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
    quant_cfg.extend(disabled_cfg)
    quant_cfg.extend(ignore_cfg)
    # The fp8 backend derives its scales from each forward's tensor, so there is nothing to calibrate.
    return {"quant_cfg": quant_cfg, "algorithm": "max" if qat_mode == "w4a16" else None}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
