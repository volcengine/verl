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
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import torch
import vllm

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
except ImportError as e:
    raise ImportError("FP8 quantization not available") from e

logger = logging.getLogger(__name__)

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


# Ref: https://github.com/NVIDIA-NeMo/RL/commit/bc24887c72a6e1b2699a228bc87c588546dfe6b7
@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])


fp8_state: FP8State = FP8State()

def validate_kv_cache_fp8_config(config):
    """
    Validate KV cache FP8 configuration to ensure both kv_cache_dtype=fp8 and 
    calculate_kv_scales=True are specified together.
    
    Args:
        config: RolloutConfig or vLLM config object
        
    Returns:
        tuple: (kv_cache_dtype, calculate_kv_scales) if valid
        
    Raises:
        ValueError: If configuration is invalid (one set without the other)
    """
    # Check for kv_cache_dtype
    kv_cache_dtype = getattr(config, 'kv_cache_dtype', None)
    if kv_cache_dtype is None:
        kv_cache_dtype = config.get('kv_cache_dtype', None) if hasattr(config, 'get') else None
    
    # Check for calculate_kv_scales
    calculate_kv_scales = getattr(config, 'calculate_kv_scales', None)
    if calculate_kv_scales is None:
        calculate_kv_scales = config.get('calculate_kv_scales', False) if hasattr(config, 'get') else False
    
    # Check if kv_cache_dtype is FP8
    is_kv_fp8 = kv_cache_dtype is not None and 'fp8' in str(kv_cache_dtype).lower()
    
    # Validate that both are set together or neither is set
    if calculate_kv_scales and not is_kv_fp8:
        raise ValueError(
            "calculate_kv_scales=True requires kv_cache_dtype to be set to fp8. "
            f"Got calculate_kv_scales={calculate_kv_scales}, kv_cache_dtype={kv_cache_dtype}"
        )
    if is_kv_fp8 and not calculate_kv_scales:
        raise ValueError(
            "kv_cache_dtype=fp8 requires calculate_kv_scales=True for dynamic scale calculation. "
            f"Got kv_cache_dtype={kv_cache_dtype}, calculate_kv_scales={calculate_kv_scales}"
        )
    
    logger.debug(f"KV cache FP8 config validated: kv_cache_dtype={kv_cache_dtype}, calculate_kv_scales={calculate_kv_scales}")
    
    return kv_cache_dtype, calculate_kv_scales

def is_kv_cache_fp8_enabled(config):
    """
    Check if FP8 KV cache with dynamic scale calculation is enabled.
    
    Args:
        config: RolloutConfig or vLLM config object
        
    Returns:
        bool: True if FP8 KV cache with calculate_kv_scales is enabled
    """
    logger.debug(f"Checking if FP8 KV cache with dynamic scale calculation is enabled: {config}")
    
    # Leverage validate_kv_cache_fp8_config to get and validate the config values
    kv_cache_dtype, calculate_kv_scales = validate_kv_cache_fp8_config(config)
    
    # Check if both FP8 KV cache dtype and dynamic scale calculation are enabled
    is_fp8_kv = (
        kv_cache_dtype is not None and 
        'fp8' in str(kv_cache_dtype).lower() and
        calculate_kv_scales
    )
    
    logger.info(
        f"kv_cache_dtype={kv_cache_dtype}, "
        f"calculate_kv_scales={calculate_kv_scales}, "
        f"is_enabled={is_fp8_kv}"
    )
    
    return is_fp8_kv


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config") and isinstance(vllm_config.quant_config, Fp8Config):
        return True

    return False


def get_module_from_param_name(model, name: str):
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if isinstance(current_module, FusedMoE):
                return current_module
            elif isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


def is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = get_module_from_param_name(model, name)
            # We currently only quantize linear layers

            if (isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn) or (
                isinstance(module, FusedMoE)
                and module.w13_weight.dtype == torch.float8_e4m3fn
                and module.w2_weight.dtype == torch.float8_e4m3fn
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


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


def quant_weights(weights, model, quant_config, dtype=torch.bfloat16):
    weights_quantized = []
    for k, v in weights:
        if not is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        if quant_config.weight_block_size is not None:
            logger.info("Using blockwise quantization")
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(dtype),
                weight_block_size=quant_config.weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            weights_quantized.append([k, param_lp])
            if vllm.__version__ >= "0.11.0":
                if "expert" in k:
                    weights_quantized.append([k + "_scale_inv", param_scale])
                else:
                    weights_quantized.append([k + "_scale", param_scale])
            else:
                weights_quantized.append([k + "_scale_inv", param_scale])

        else:
            raise ValueError(
                "Currently only support blockwise quantization, please set weight_block_size in quant_config"
            )

    return weights_quantized


def load_quanted_weights(weights, model_runner):
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config
    vllm_dtype = model_runner.vllm_config.model_config.dtype

    weights_quantized = quant_weights(weights, model, quant_config, dtype=vllm_dtype)

    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    loaded_params = model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type
    return loaded_params


def process_weights_after_loading_for_vllm10(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm v0.10

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    logger.debug("Applying patch process_weights_after_loading")
    try:
        from vllm.model_executor.parameter import (
            BlockQuantScaleParameter,
            ModelWeightParameter,
        )
    except Exception:
        print("error")
    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_param_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"
    weight = layer.weight.data
    weight_scale_inv = layer.weight_scale_inv.data
    weight = self._maybe_pad_weight(weight)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )


def process_weights_after_loading_for_vllm11(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm 0.11

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_param_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    weight_scale = layer.weight_scale_inv if hasattr(layer, "weight_scale_inv") else layer.weight_scale
    weight, weight_scale = process_fp8_weight_block_strategy(layer.weight, weight_scale)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )

    del layer.weight_scale_inv

    # invoke with just 1 parameter for vllm v0.11.1 and above
    if vllm.__version__ >= "0.11.1":
        maybe_post_process_fp8_weight_block(layer)
    else:
        maybe_post_process_fp8_weight_block(layer, self.cutlass_block_fp8_supported)


def process_weights_after_loading_moe_for_vllm10(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer, it is used for vllm v0.10"""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled
    from vllm.model_executor.layers.quantization.fp8 import _is_col_major, _swap_w13_to_w31
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_col_major_tma_aligned_tensor,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import is_blackwell_deep_gemm_used

    self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
    assert self.quant_config.activation_scheme == "dynamic"
    if self.flashinfer_moe_enabled:
        w13_weight = _swap_w13_to_w31(layer.w13_weight.data)
        w13_weight_scale_inv = _swap_w13_to_w31(layer.w13_weight_scale_inv.data)
        w2_weight = layer.w2_weight.data
        w2_weight_scale_inv = layer.w2_weight_scale_inv.data
    else:
        w13_weight = layer.w13_weight.data
        w13_weight_scale_inv = layer.w13_weight_scale_inv.data
        w2_weight = layer.w2_weight
        w2_weight_scale_inv = layer.w2_weight_scale_inv

    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_data, custom_weight):
        param = Parameter(custom_data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_weight_dir = dir(custom_weight)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_weight_dir if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_weight, attr))

        return param

    layer.w13_weight = _create_param_from_subclass_attributes(w13_weight, layer.w13_weight)
    layer.w13_weight_scale_inv = _create_param_from_subclass_attributes(
        w13_weight_scale_inv, layer.w13_weight_scale_inv
    )
    layer.w2_weight = _create_param_from_subclass_attributes(w2_weight, layer.w2_weight)
    layer.w2_weight_scale_inv = _create_param_from_subclass_attributes(w2_weight_scale_inv, layer.w2_weight_scale_inv)

    # DeepGemm scales need to be transposed and aligned.  We try to do
    # it ahead of time for performance reasons.
    if self.allow_deep_gemm and not is_blackwell_deep_gemm_used():
        # Lazy import to avoid CUDA initialization problems.
        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()

    if is_blackwell_deep_gemm_used():
        assert layer.weight_block_size is not None
        # Re-quantise the expert weights so their scales are UE8M0.
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_sz,
        )
        requant_weight_ue8m0_inplace(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_sz,
        )

        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()


def process_weights_after_loading_moe_for_vllm11(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer, it is used for vllm 0.11"""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        expert_weight_is_col_major,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import (
        get_col_major_tma_aligned_tensor,
        is_deep_gemm_e8m0_used,
    )

    self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_backend is not None:
        layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        layer.w13_weight_scale_inv.data = swap_w13_to_w31(layer.w13_weight_scale_inv.data)

    if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)

    if is_deep_gemm_e8m0_used():
        assert layer.weight_block_size is not None
        # Re-quantise the expert weights so their scales are UE8M0.
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_sz,
        )
        requant_weight_ue8m0_inplace(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_sz,
        )

        # Ensure column-major TMA alignment expected by DeepGEMM.
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)


def reset_kv_scale_flags_in_model(model_runner) -> int:
    """
    Reset calculate_kv_scales flags in model_runner and all attention layers.
    Also restore range constants (q_range, k_range, v_range) that may have been
    corrupted during wake_up.
    
    This should be called after loading new weights to trigger KV scale recalculation
    on the next forward pass.
    Args:
        model_runner: vLLM ModelRunner instance
        
    Returns:
        int: Number of attention layers where flags were reset
    """
    logger.info(f"Resetting KV scale calculation flags in model_runner: {model_runner}")
    
    # Import vllm envs to get the scale constants
    from vllm import envs
    
    # Reset model_runner level flag
    if hasattr(model_runner, 'calculate_kv_scales'):
        model_runner.calculate_kv_scales = True
        logger.debug("Reset model_runner.calculate_kv_scales = True")
    else:
        logger.debug("model_runner does not have calculate_kv_scales attribute")
    
    # Reset per-layer flags and restore range constants
    model = model_runner.model
    num_reset = 0
    num_ranges_restored = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'calculate_kv_scales'):
            module.calculate_kv_scales = True
            num_reset += 1
            
            # CRITICAL FIX: Restore range constants that get corrupted during wake_up
            # These are used as denominators in scale calculation: scale = max_value / range
            # If range becomes 0 or garbage, we get inf/nan
            ranges_restored = False
            if hasattr(module, 'q_range') and module.q_range is not None:
                try:
                    module.q_range.data.fill_(envs.Q_SCALE_CONSTANT)
                    ranges_restored = True
                except Exception as e:
                    logger.warning(f"Failed to restore q_range for {name}: {e}")
            
            if hasattr(module, 'k_range') and module.k_range is not None:
                try:
                    module.k_range.data.fill_(envs.K_SCALE_CONSTANT)
                    ranges_restored = True
                except Exception as e:
                    logger.warning(f"Failed to restore k_range for {name}: {e}")
            
            if hasattr(module, 'v_range') and module.v_range is not None:
                try:
                    module.v_range.data.fill_(envs.V_SCALE_CONSTANT)
                    ranges_restored = True
                except Exception as e:
                    logger.warning(f"Failed to restore v_range for {name}: {e}")
            
            if ranges_restored:
                num_ranges_restored += 1
    
    if num_reset > 0:
        logger.info(f"Reset calculate_kv_scales flag in {num_reset} attention layers")
        logger.info(f"Restored range constants (q/k/v_range) in {num_ranges_restored} layers (fixes inf/nan)")
    else:
        logger.warning("No attention layers with calculate_kv_scales found")
    
    return num_reset


def patched_maybe_post_process_fp8_weight_block(layer: torch.nn.Module):
    """Patched version that preserves parameter attributes and types.
    
    This patches vllm's maybe_post_process_fp8_weight_block function which
    creates new Parameter objects for DeepGEMM optimization but loses:
    - The parameter class type (ModelWeightParameter, BlockQuantScaleParameter)
    - All custom attributes (weight_loader, output_dim, input_dim, etc.)
    - All custom methods (load_qkv_weight, load_column_parallel_weight, etc.)
    
    Instead of creating new Parameters, we update the existing parameter data
    in-place, which preserves everything needed for weight reloading.
    """
    assert layer.weight_block_size is not None

    from vllm.utils.deep_gemm import (
        is_deep_gemm_e8m0_used,
        should_use_deepgemm_for_fp8_linear,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    
    # On Blackwell or Hopper, if E8M0 for DeepGemm is used, we need to
    # requantize the weight and input to the specific scale at the same time.
    should_use_deepgemm = should_use_deepgemm_for_fp8_linear(
        layer.orig_dtype, layer.weight
    )
    if should_use_deepgemm:
        dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=layer.weight.data,
            ws=layer.weight_scale.data,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
        # CRITICAL: Update data in-place to preserve parameter type and ALL attributes
        # Do NOT create new Parameter objects as that loses:
        # - Parameter subclass types (ModelWeightParameter, BlockQuantScaleParameter)
        # - Custom attributes (weight_loader, output_dim, input_dim, etc.)
        # - Custom methods (load_qkv_weight, load_column_parallel_weight, etc.)
        layer.weight.data.copy_(dg_weight)
        layer.weight_scale.data.copy_(dg_weight_scale)


def apply_vllm_fp8_patches():
    """
    Apply all vLLM FP8 patches for blockwise quantization.
    """
    logger.info("Applying vllm fp8 patches for blockwise quantization")
    func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
    patcher1 = patch(
        func1_path,
        process_weights_after_loading_for_vllm11
        if vllm.__version__ >= "0.11.0"
        else process_weights_after_loading_for_vllm10,
    )
    patcher1.start()
    func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"
    patcher2 = patch(
        func2_path,
        process_weights_after_loading_moe_for_vllm11
        if vllm.__version__ >= "0.11.0"
        else process_weights_after_loading_moe_for_vllm10,
    )
    patcher2.start()
    
    # CRITICAL: Patch maybe_post_process_fp8_weight_block to preserve parameter types and attributes
    # This is needed for v0.11.1+ where DeepGEMM optimization would create new Parameters,
    # losing all custom attributes, methods, and the parameter subclass type preventing weight reloading.
    # Ref: https://github.com/vllm-project/vllm/pull/27897/files
    if vllm.__version__ >= "0.11.1":
        func3_path = "vllm.model_executor.layers.quantization.utils.fp8_utils.maybe_post_process_fp8_weight_block"
        patcher3 = patch(func3_path, patched_maybe_post_process_fp8_weight_block)
        patcher3.start()
        logger.info("Applied patch for maybe_post_process_fp8_weight_block to preserve parameter types/attributes")
