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
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

import torch

try:
    from vllm._custom_ops import scaled_fp8_quant
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
            if isinstance(current_module, torch.nn.ModuleList):
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
            if isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn:
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


def quant_weights(weights, model, quant_config):
    weights_quantized = []
    for k, v in weights:
        if not is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        if quant_config.weight_block_size is not None:
            logger.info("Using blockwise quantization")
            param_lp, param_scale = scaled_fp8_blockwise(
                v.to(torch.float),
                weight_block_size=quant_config.weight_block_size,
            )
            param_scale = param_scale.squeeze(-1)
            weights_quantized.append([k, param_lp])
            weights_quantized.append([k + "_scale_inv", param_scale])

        else:
            logger.info("Using Per tensor quantization")
            original_shape = v.shape
            # Use per tensor quantization
            quantized_tensor, scale = scaled_fp8_quant(v)
            # Reshape back to original shape
            quantized_tensor = quantized_tensor.view(original_shape)

            scale_k = k.replace(".weight", ".weight_scale")
            scale = scale.view(1)
            weights_quantized.extend([(k, quantized_tensor), (scale_k, scale)])

    return weights_quantized


def load_quanted_weights(weights, model_runner):
    model = model_runner.model
    quant_config = model_runner.vllm_config.quant_config

    weights_quantized = quant_weights(weights, model, quant_config)

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


def process_weights_after_loading(self, layer) -> None:
    try:
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale
        from vllm.model_executor.parameter import (
            BlockQuantScaleParameter,
            ModelWeightParameter,
            PerTensorScaleParameter,
        )
    except Exception:
        try:
            from sglang.srt.layers.parameter import (
                BlockQuantScaleParameter,
                ModelWeightParameter,
                PerTensorScaleParameter,
            )
            from sglang.srt.layers.quantization.utils import requantize_with_max_scale
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

    if self.block_quant:
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

    else:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        # # If using w8a8, torch._scaled_mm needs per tensor, so
        # # requantize the logical shards as a single weight.
        if not self.use_marlin:
            # Dequant -> Quant with max scale so we can run per tensor.

            weight_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=weight_scale,
                logical_widths=layer.logical_widths,
            )

        weight = self._maybe_pad_weight(weight)
        # Update layer with new values.
        # layer.weight = Parameter(weight.t(), requires_grad=False)
        # layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        layer.weight = _create_param_from_subclass_attributes(
            ModelWeightParameter(
                data=weight,
                output_dim=0,
                input_dim=1,
                weight_loader=layer.weight.weight_loader,
            )
        )
        layer.weight_scale = _create_param_from_subclass_attributes(
            PerTensorScaleParameter(
                data=weight_scale.repeat(len(layer.logical_widths)),
                weight_loader=layer.weight_scale.weight_loader,
            )
        )


def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

    if self.use_marlin:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    if self.block_quant:
        assert self.quant_config.weight_block_size is not None
        return torch.ops.vllm.apply_w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
            use_aiter_and_is_supported=self.use_aiter_and_is_supported,
        )

    weight_scale, weight = requantize_with_max_scale(
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        logical_widths=layer.logical_widths,
    )
    return self.fp8_linear.apply(
        input=x,
        weight=weight.t(),
        weight_scale=weight_scale,
        out_dtype=self.out_dtype,
        input_scale=layer.input_scale,
        bias=bias,
    )


def apply_vllm_fp8_patches(block_quant=True):
    if block_quant:
        func_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher = patch(func_path, process_weights_after_loading)
        patcher.start()
    else:
        func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher1 = patch(func1_path, process_weights_after_loading)
        patcher1.start()
        func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.apply"
        patcher2 = patch(func2_path, apply)
        patcher2.start()
