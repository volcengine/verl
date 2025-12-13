# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from torchao.core.config import AOBaseConfig
from .kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import is_sm_at_least_90


class fp8_blockwise_mm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size, out_dtype=torch.bfloat16, use_triton=False):
        assert block_size == 128, "Only support block_size=128"

        # 1. Reshape to 2D [Tokens, Hidden]
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])
        
        M, K = x.shape
        
        # 2. Dynamic padding logic: pad to multiples of 128
        pad_m = (block_size - M % block_size) % block_size
        if pad_m > 0:
            # Pad pad_m rows of zeros at the bottom (dimension 0)
            x_padded = F.pad(x, (0, 0, 0, pad_m))
        else:
            x_padded = x

        # 3. Quantization (using padded data)
        # Cast inputs to fp8 blockwise using (1, block_size) scaling granularity in row major format.
        x_fp8, x_scale = triton_fp8_blockwise_act_quant_lhs(x_padded, block_size)

        # Cast weight to fp8 blockwise using (block_size, block_size) scaling granularity, with transposed dims in column major format.
        weight_t_fp8, weight_t_scale = triton_fp8_blockwise_weight_quant_transposed_rhs(
            weight,
            block_size=block_size,
        )

        # 4. Compute GEMM
        fp8_gemm = triton_fp8_gemm_1x128_128x128 if use_triton else torch._scaled_mm
        out_padded = fp8_gemm(
            x_fp8,
            weight_t_fp8,
            x_scale,
            weight_t_scale,
            out_dtype=out_dtype,
        )
        
        # 5. Unpadding: remove the padded zeros
        if pad_m > 0:
            out = out_padded[:M, :]
        else:
            out = out_padded

        out = out.reshape(*x_orig_shape[:-1], out.shape[-1])
        
        # Save context, note that we need to save pad_m for backward pass
        ctx.save_for_backward(x, weight)
        ctx.block_size = block_size
        ctx.out_dtype = out_dtype
        ctx.use_triton = use_triton
        ctx.pad_m = pad_m  # Save for backward
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        block_size = ctx.block_size
        out_dtype = ctx.out_dtype
        use_triton = ctx.use_triton
        pad_m = ctx.pad_m  # Retrieve for backward

        # 1. Reshape Input & Grad to 2D
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        grad_output_orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output_orig_shape[-1]).contiguous()
        
        # 2. If we padded in forward, we also need to pad in backward!
        # Because Triton kernel requires grad_output rows to be multiples of 128
        if pad_m > 0:
            grad_output_padded = F.pad(grad_output, (0, 0, 0, pad_m))
            x_padded = F.pad(x, (0, 0, 0, pad_m))  # x also needs to be padded again for dW calculation
        else:
            grad_output_padded = grad_output
            x_padded = x

        # All subsequent computations use _padded variables
        
        # Cast grad_output to fp8 blockwise 1x128 since it is the grad of the output activation.
        grad_output_fp8, grad_output_scale = triton_fp8_blockwise_act_quant_lhs(
            grad_output_padded,
            block_size,
        )

        # Cast weight to fp8 blockwise to 128x128 in column major format.
        weight_fp8, weight_scale = triton_fp8_blockwise_weight_quant_rhs(
            weight,
            block_size=block_size,
        )

        # grad_x = grad_output @ weight
        fp8_gemm_1x128_128x128 = (
            triton_fp8_gemm_1x128_128x128 if use_triton else torch._scaled_mm
        )
        grad_x_padded = fp8_gemm_1x128_128x128(
            grad_output_fp8,
            weight_fp8,
            grad_output_scale,
            weight_scale,
            out_dtype=out_dtype,
        )

        # Cast grad_output_t to fp8 blockwise with (1 x block_size) scaling groups, since it is
        # the grad of the output activation.
        # Write directly with transposed dims in row major format, as needed for dW calc.
        grad_output_t_fp8, grad_output_t_scale = (
            triton_fp8_blockwise_act_quant_transposed_lhs(
                grad_output_padded,
                block_size,
            )
        )

        # Cast x to fp8 blockwise with (block_size x 1) scaling groups, in column major format.
        # RHS should have groupwise scales calculated colwise, so scaling groups do not cross the
        # contracting (K) dim.
        x_fp8, x_scale = triton_fp8_blockwise_act_quant_rhs(x_padded, block_size)

        # grad_weight = grad_output.T @ x
        fp8_gemm_1x128_128x1 = (
            triton_fp8_gemm_1x128_128x1 if use_triton else torch._scaled_mm
        )
        grad_weight = fp8_gemm_1x128_128x1(
            grad_output_t_fp8,
            x_fp8,
            grad_output_t_scale,
            x_scale,
            out_dtype=out_dtype,
        )

        # 3. Unpad grad_x
        if pad_m > 0:
            # Remove the padded rows to restore the original gradient shape
            grad_x = grad_x_padded[:grad_output.shape[0], :]
        else:
            grad_x = grad_x_padded

        # Reshape grad_x to expected potentially 3D+ shape
        grad_x = grad_x.reshape(*grad_output_orig_shape[:-1], grad_x.shape[-1])
        return grad_x, grad_weight, None, None, None


class Float8BlockwiseLinear(nn.Linear):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
        dtype (torch.dtype): Data type for the weights. Defaults to torch.float8_e4m3fn.
    """

    supported_dtypes = [
        torch.bfloat16,
    ]

    def __init__(
        self,
        *args,
        block_size: int = 128,
        dtype=torch.bfloat16,
        use_triton=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert dtype in self.supported_dtypes, (
            f"Unsupported dtype: {dtype}. Supported dtypes: {self.supported_dtypes}"
        )
        assert is_sm_at_least_90(), "Only support SM90"
        self.block_size = block_size
        self.dtype = dtype
        self.use_triton = use_triton
        
        # Ensure weight dtype matches the specified dtype
        # nn.Linear may initialize weight with default dtype (float32),
        # so we need to convert it to the target dtype
        if self.weight.dtype != dtype:
            self.weight.data = self.weight.data.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return fp8_blockwise_mm.apply(
            x, self.weight, self.block_size, self.dtype, self.use_triton
        )

    @classmethod
    def from_float(
        cls,
        mod,
    ):
        assert mod.bias is None, "unsupported"
        assert mod.in_features % 128 == 0, "unsupported"
        assert mod.out_features % 128 == 0, "unsupported"
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8BlockwiseLinearConfig(AOBaseConfig):
    pass


@register_quantize_module_handler(Float8BlockwiseLinearConfig)
def _float8_blockwise_transform(module, config):
    return Float8BlockwiseLinear.from_float(module)


def recursive_replace_blockwise(
    module: nn.Module,
    block_size: int = 128,
    target_dtype: torch.dtype = torch.bfloat16,
    use_triton: bool = True,
    exclude_modules: list[str] | None = None,
    require_no_bias: bool = True,
) -> None:
    """
    Recursively replace all nn.Linear layers in a module with Float8BlockwiseLinear.
    
    Args:
        module: The root module to process.
        block_size: Block size for quantization. Defaults to 128.
        target_dtype: Data type for the weights. Defaults to torch.bfloat16.
        use_triton: Whether to use Triton kernels. Defaults to True.
        exclude_modules: List of module names to exclude from replacement (e.g., ["lm_head"]).
        require_no_bias: If True, only replace Linear layers without bias. Defaults to True.
    """
    if exclude_modules is None:
        exclude_modules = []
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Skip excluded modules
            if any(excluded in name for excluded in exclude_modules):
                continue
            
            # Skip layers with bias if require_no_bias is True
            if require_no_bias and child.bias is not None:
                continue
            
            # Get device and dtype from the original layer
            target_device = child.weight.device
            
            # Create new Float8BlockwiseLinear layer
            new_layer = Float8BlockwiseLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=False,
                block_size=block_size,
                device=target_device,
                dtype=target_dtype,
                use_triton=use_triton,
            )
            
            new_layer.weight.data = child.weight.data.to(dtype=target_dtype)

            # Replace the layer
            setattr(module, name, new_layer)
        else:
            # Recursively process child modules
            recursive_replace_blockwise(
                child,
                block_size=block_size,
                target_dtype=target_dtype,
                use_triton=use_triton,
                exclude_modules=exclude_modules,
                require_no_bias=require_no_bias,
            )
