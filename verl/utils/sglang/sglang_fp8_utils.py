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

import logging

import torch

logger = logging.getLogger(__name__)


def should_quantize_param(param_name: str) -> bool:
    """根据参数名判断是否应该量化为 FP8

    量化规则：
    - 必须以 .weight 结尾（排除 bias）
    - 排除 embedding 层
    - 排除 normalization 层
    - 排除输出层（lm_head）
    """
    # 必须是权重参数
    if not param_name.endswith(".weight"):
        return False

    # 排除的层类型
    exclude_patterns = [
        "embed_tokens",  # Embedding 层
        "lm_head",  # 输出层
        "layernorm",  # LayerNorm
        "norm",  # 各种 Norm 层
        "ln_",  # LayerNorm 变体
        "embeddings",  # Embeddings
    ]

    # 检查是否匹配排除模式
    param_lower = param_name.lower()
    for pattern in exclude_patterns:
        if pattern in param_lower:
            return False

    # 包含的层类型（Linear 层）
    include_patterns = [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # Gate projection (for MLP)
        "up_proj",  # Up projection (for MLP)
        "down_proj",  # Down projection (for MLP)
        "fc1",  # Fully connected 1
        "fc2",  # Fully connected 2
        "gate",  # Gate (for MoE)
        "mlp",  # MLP layers
    ]

    # 检查是否匹配包含模式
    for pattern in include_patterns:
        if pattern in param_lower:
            logger.info(f"Will quantize FP8: {param_name}")
            return True

    # 默认不量化
    logger.debug(f"Skip quantization: {param_name}")
    return False


def quant_weights_by_name(weights, quant_config, dtype=torch.bfloat16, vllm_version="0.11.0"):
    """基于参数名的 FP8 量化

    Args:
        weights: Generator of (name, tensor) pairs
        quant_config: Quantization configuration
        dtype: Data type for intermediate computation
        vllm_version: vLLM version string for scale naming

    Returns:
        List of (name, tensor) pairs with quantized weights
    """
    from verl.utils.sglang.sglang_fp8_utils import scaled_fp8_blockwise

    weights_quantized = []
    quantized_count = 0
    skipped_count = 0

    if isinstance(quant_config, dict):
        weight_block_size = quant_config.get("weight_block_size")
    else:
        weight_block_size = getattr(quant_config, "weight_block_size", None)

    if weight_block_size is None:
        raise ValueError("weight_block_size not found in quant_config")

    for k, v in weights:
        # 判断是否需要量化
        if not should_quantize_param(k):
            weights_quantized.append((k, v))
            skipped_count += 1
            continue

        # 量化为 FP8
        try:
            if weight_block_size is not None:
                print(f"Quantizing to FP8 blockwise: {k}")
                param_lp, param_scale = scaled_fp8_blockwise(
                    v.to(dtype),
                    weight_block_size=weight_block_size,
                )
                param_scale = param_scale.squeeze(-1)
                weights_quantized.append([k, param_lp])
                weights_quantized.append([k + "_scale_inv", param_scale])
                quantized_count += 1
            else:
                raise ValueError(
                    "Only blockwise quantization is supported. Please set weight_block_size in quant_config"
                )
        except Exception as e:
            logger.error(f"Failed to quantize {k}: {e}")
            # 如果量化失败，使用原始权重
            weights_quantized.append((k, v))
            skipped_count += 1

    print(f"FP8 quantization complete: {quantized_count} quantized, {skipped_count} skipped")
    return weights_quantized


def scaled_fp8_blockwise(
    data_hp,
    weight_block_size,
):
    # cast tensor from high precision to FP8 with 128*128 blockwise quantization.
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    )
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
    )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)

    # Use FP32 scale
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    # preserve the behavior for 0 amax case
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    descale_fp = torch.reciprocal(scale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(original_shape)

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp
