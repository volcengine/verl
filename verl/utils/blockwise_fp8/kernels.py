# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# from torchao.prototype.moe_training.utils import (
#     _is_column_major,
#     _is_row_major,
# )

def _is_column_major(x: torch.Tensor) -> bool:
    """
    This function checks if the input tensor is column-major.

    Args:
        x (torch.Tensor): The input tensor to be checked.

    Returns:
        A boolean indicating whether the input tensor is column-major.
    """
    assert x.ndim == 2 or x.ndim == 3, "input tensor must be 2D or 3D"
    return x.stride(-2) == 1


def _is_row_major(x: torch.Tensor) -> bool:
    """
    This function checks if the input tensor is row-major.

    Args:
        x (torch.Tensor): The input tensor to be checked.

    Returns:
        A boolean indicating whether the input tensor is row-major.
    """
    assert x.ndim == 2 or x.ndim == 3, "input tensor must be 2D or 3D"
    return x.stride(-1) == 1

fp8_gemm_configs_max_autotune = [
    triton.Config(
        {"BLOCK_SIZE_M": block_size, "BLOCK_SIZE_N": block_size},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_size in [64, 128, 256]
    for num_warps in [4, 8]
    for num_stages in [2]
]

EPS = 1e-12


@triton.autotune(configs=fp8_gemm_configs_max_autotune, key=["N", "K", "BLOCK_SIZE_K"])
@triton.jit
def triton_fp8_gemm_1x128_128x128_kernel(
    a_ptr,  # (M, K)
    a_stride_dim_0,
    a_stride_dim_1,
    b_ptr,  # (K, N)
    b_stride_dim_0,
    b_stride_dim_1,
    c_ptr,
    c_stride_dim_0,
    c_stride_dim_1,
    a_s_ptr,  # (M, K // block_size) reciprocals of scales
    a_s_stride_dim_0,
    a_s_stride_dim_1,
    b_s_ptr,  # (K // block_size, N // block_size) reciprocals of scales
    b_s_stride_dim_0,
    b_s_stride_dim_1,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_m[:, None] * a_stride_dim_0 + offs_k[None, :] * a_stride_dim_1
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * b_stride_dim_0 + offs_n[None, :] * b_stride_dim_1
    )

    k_num_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    # Scale base pointers start at row offsets for A, and column offsets for B.
    a_s_base_ptr = a_s_ptr + offs_m * a_s_stride_dim_0
    b_s_base_ptr = b_s_ptr + (offs_n // BLOCK_SIZE_K) * b_s_stride_dim_1
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    for k in range(0, k_num_blocks):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Reciprocal scales to scale back to dynamic range of output dtype
        a_s = tl.load(a_s_base_ptr + k * a_s_stride_dim_1)
        b_s = tl.load(b_s_base_ptr + k * b_s_stride_dim_0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s

        a_ptrs += BLOCK_SIZE_K * a_stride_dim_1
        b_ptrs += BLOCK_SIZE_K * b_stride_dim_0

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * c_stride_dim_0 + offs_n[None, :] * c_stride_dim_1
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton_op("torchao::triton_fp8_gemm_1x128_128x128", mutates_args={})
def triton_fp8_gemm_1x128_128x128(
    a: torch.Tensor,  # (M, K)
    b: torch.Tensor,  # (K, N)
    a_s: torch.Tensor,  # (M, K // block_size)
    b_s: torch.Tensor,  # (K // block_size, N // block_size)
    block_size: int = 128,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # 'a' must be in row-major layout, 'b' must be in column-major layout
    assert _is_row_major(a), "a must be row-major"
    assert _is_column_major(b), "b must be column-major"

    # a_scales must be col-major, b_scales must be row-major
    assert _is_column_major(a_s), "a_s must be column-major"
    assert _is_column_major(b_s), "b_s must be column-major"

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)
    c = a.new_empty(M, N, dtype=out_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    wrap_triton(triton_fp8_gemm_1x128_128x128_kernel)[grid](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        c.stride(0),
        c.stride(1),
        a_s,
        a_s.stride(0),
        a_s.stride(1),
        b_s,
        b_s.stride(0),
        b_s.stride(1),
        M,
        N,
        K,
        out_dtype=out_dtype,
        BLOCK_SIZE_K=block_size,
    )
    return c


@triton.autotune(
    configs=fp8_gemm_configs_max_autotune, key=["M", "N", "K", "BLOCK_SIZE_K"]
)
@triton.jit
def triton_fp8_gemm_1x128_128x1_kernel(
    a_ptr,  # (M, K)
    a_stride_dim_0,
    a_stride_dim_1,
    b_ptr,  # (K, N)
    b_stride_dim_0,
    b_stride_dim_1,
    c_ptr,
    a_s_ptr,  # (M, K // block_size)
    a_s_stride_dim_0,
    a_s_stride_dim_1,
    b_s_ptr,  # (K // block_size, N)
    b_s_stride_dim_0,
    b_s_stride_dim_1,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_m[:, None] * a_stride_dim_0 + offs_k[None, :] * a_stride_dim_1
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * b_stride_dim_0 + offs_n[None, :] * b_stride_dim_1
    )

    k_num_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    a_s_base_ptr = a_s_ptr + offs_m * a_s_stride_dim_0
    b_s_base_ptr = b_s_ptr + offs_n * b_s_stride_dim_1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, k_num_blocks):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Reciprocal scales to scale back to dynamic range of output dtype
        a_s = tl.load(a_s_base_ptr + k * a_s_stride_dim_1)
        b_s = tl.load(b_s_base_ptr + k * b_s_stride_dim_0)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_SIZE_K * a_stride_dim_1
        b_ptrs += BLOCK_SIZE_K * b_stride_dim_0

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton_op("torchao::triton_fp8_gemm_1x128_128x1", mutates_args={})
def triton_fp8_gemm_1x128_128x1(
    a: torch.Tensor,  # (M, K)
    b: torch.Tensor,  # (K, N)
    a_s: torch.Tensor,  # (M, K // block_size) reciprocals of scales
    b_s: torch.Tensor,  # (K // block_size, N) reciprocals of scales
    block_size: int = 128,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # 'a' must be in row-major layout, 'b' must be in column-major layout
    assert _is_row_major(a), "a must be row-major"
    assert _is_column_major(b), "b must be column-major"

    # a_scales must be col-major, b_scales must be row-major
    assert _is_column_major(a_s), "a_s must be column-major"
    assert _is_row_major(b_s), "b_s must be row-major"

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)
    c = a.new_empty(M, N, dtype=out_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    wrap_triton(triton_fp8_gemm_1x128_128x1_kernel)[grid](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        a_s,
        a_s.stride(0),
        a_s.stride(1),
        b_s,
        b_s.stride(0),
        b_s.stride(1),
        M,
        N,
        K,
        BLOCK_SIZE_K=block_size,
    )
    return c


# Quantization kernels autotuner configs
quant_kernel_configs = [
    triton.Config(
        {},
        num_warps=warps,
        num_stages=stages,
    )
    for warps in [4, 8]
    for stages in [2, 4]
]

quant_kernel_configs_with_groups = [
    triton.Config(
        {"NUM_GROUPS": groups},
        num_warps=warps,
        num_stages=stages,
    )
    for groups in [2, 16, 32, 64, 128]
    for warps in [2, 4, 8]
    for stages in [2, 4, 6]
]


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["K"])
@triton.jit
def triton_fp8_blockwise_act_quant_lhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Load (num_groups x block_size) tile of x, where input is row major
    m_offs = pid_m * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    k_offs = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Perform scaling
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0

    # Scales for (1 x block_size) groups, shape will be (NUM_GROUPS, 1)
    amax = tl.clamp(tl.max(tl.abs(x), axis=1), min=EPS, max=float("inf")).to(tl.float64)
    scale = (max_fp8_e4m3 / amax).to(tl.float32)[:, None]
    y = x * scale
    y = tl.clamp(y, min=min_fp8_e4m3, max=max_fp8_e4m3).to(y_ptr.dtype.element_ty)

    # Write output to column major fomrat
    y_offs = m_offs[:, None] * y_stride_dim_0 + k_offs[None, :] * y_stride_dim_1
    y_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    tl.store(y_ptr + y_offs, y, mask=y_mask)

    # Write reciprocal scales
    scale_offs = m_offs[:, None] * s_stride_dim_0 + pid_k * s_stride_dim_1
    tl.store(s_ptr + scale_offs, tl.div_rn(1.0, scale))


@triton_op("torchao::triton_fp8_blockwise_act_quant_lhs", mutates_args={})
def triton_fp8_blockwise_act_quant_lhs(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: row-major high-precision tensor
    Output: row-major, with reciprocal scales for (1 x block_size) groups stored in col-major.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
    ], "dtype must be torch.float8_e4m3fn"
    M, K = x.size()
    y = torch.empty_like(x, dtype=dtype)
    # Write scales to column-major format to align with torch._scaled_mm requirements.
    s = x.new_empty(M, K // block_size, dtype=torch.float32).as_strided(
        (M, K // block_size),
        (1, M),
    )

    def grid(meta):
        return (
            triton.cdiv(M, meta["NUM_GROUPS"]),
            triton.cdiv(K, meta["BLOCK_SIZE"]),
        )

    wrap_triton(triton_fp8_blockwise_act_quant_lhs_kernel)[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M,
        K=K,
        BLOCK_SIZE=block_size,
        EPS=EPS,
    )
    return y, s


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["K"])
@triton.jit
def triton_fp8_blockwise_act_quant_rhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Load (block_size x block_size) tile of x, where input is row major.
    # Each scaling group is (block_size x 1), but we load (block_size x block_size)
    # to facilitate coalesced gmem accesses and improve efficiency.
    m_offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_offs = pid_k * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Perform scaling
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0

    # Column-wise scales for RHS operand, shape (1, block_size)
    amax = tl.clamp(tl.max(tl.abs(x), axis=0), min=EPS, max=float("inf")).to(tl.float64)
    scale = (max_fp8_e4m3 / amax).to(tl.float32)[None, :]
    y = x * scale
    y = tl.clamp(y, min=min_fp8_e4m3, max=max_fp8_e4m3).to(y_ptr.dtype.element_ty)

    # Write output to column major format
    y_offs = m_offs[:, None] * y_stride_dim_0 + k_offs[None, :] * y_stride_dim_1
    y_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    tl.store(y_ptr + y_offs, y, mask=y_mask)

    # Write scales
    scale_offs = pid_m * s_stride_dim_0 + k_offs[None, :] * s_stride_dim_1
    tl.store(s_ptr + scale_offs, tl.div_rn(1.0, scale))


@triton_op("torchao::triton_fp8_blockwise_act_quant_rhs", mutates_args={})
def triton_fp8_blockwise_act_quant_rhs(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: row-major
    Output: column-major, with scales for (block_size x 1) groups stored in row-major.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
    ], "dtype must be torch.float8_e4m3fn"
    M, K = x.size()
    M_blocks = triton.cdiv(M, block_size)
    y = torch.empty_like(x, dtype=dtype)
    y = y.as_strided(y.size(), (1, y.size(0)))
    s = x.new_empty(M_blocks, K, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(K, meta["NUM_GROUPS"]),
        )

    wrap_triton(triton_fp8_blockwise_act_quant_rhs_kernel)[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M=M,
        K=K,
        BLOCK_SIZE=block_size,
        EPS=EPS,
    )
    return y, s


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["K"])
@triton.jit
def triton_fp8_blockwise_act_quant_transposed_lhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # For scaling groups, not for grid/parallelization
    NUM_GROUPS: tl.constexpr,  # For grid/parallelization, not for scaling groups
    EPS: tl.constexpr,
):
    # This kernel reads data in row-major format, and writes to an output tensor with
    # transposed dims and in column major format. To facilitate this, given that for a
    # LHS operator the scales must be rowwise, we will be computing colwise scales on the
    # original data, then writing the scaled data rowwise.
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Load (block_size x num_groups) block of input, where input is row major.
    # We will be computing (block_size x 1) scaling factors (columns), and computing
    # `num_groups` at a time, so we aren't parallelizing with 1 thread per column,
    # which will fail to launch for large tensors, due to max block number of 65535.
    m_offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_offs = pid_k * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Perform scaling
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0

    # Compute amax across dim 0 (column-wise).
    amax = tl.clamp(tl.max(tl.abs(x), axis=0), min=EPS, max=float("inf")).to(tl.float64)
    scale = (max_fp8_e4m3 / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=min_fp8_e4m3, max=max_fp8_e4m3).to(y_ptr.dtype.element_ty)

    # Write output to column major fomrat
    y_offs = k_offs[:, None] * y_stride_dim_0 + m_offs[None, :] * y_stride_dim_1
    y_mask = (k_offs[:, None] < K) & (m_offs[None, :] < M)
    tl.store(y_ptr + y_offs, y.trans(1, 0), mask=y_mask)

    # Scales are one per column (block_size x 1).
    scale_m_off = pid_m
    scale_k_offs = k_offs

    # Scale tensor size is (K, M // SCALE_BLOCK_SIZE)
    scale_offs = scale_k_offs * s_stride_dim_0 + scale_m_off * s_stride_dim_1
    scale_mask = (scale_k_offs < K) & (scale_m_off < M // BLOCK_SIZE)

    # Write out reciprocal scales
    tl.store(s_ptr + scale_offs, tl.div_rn(1.0, scale), mask=scale_mask)


@triton_op("torchao::triton_fp8_blockwise_act_quant_transposed_lhs", mutates_args={})
def triton_fp8_blockwise_act_quant_transposed_lhs(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(0) % block_size == 0, (
        f"First dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
    ], "dtype must be torch.float8_e4m3fn"

    # Output should have transposed dims and be in row major format
    M, K = x.shape
    y = torch.empty(K, M, dtype=dtype, device=x.device)
    M_blocks = triton.cdiv(M, block_size)

    # Column major scales required for torch._scaled_mm
    s = x.new_empty(K, M_blocks, dtype=torch.float32).as_strided(
        (K, M_blocks),  # shape
        (1, K),  # stride
    )

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(K, meta["NUM_GROUPS"]),
        )

    wrap_triton(triton_fp8_blockwise_act_quant_transposed_lhs_kernel)[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M,
        K=K,
        BLOCK_SIZE=block_size,  # Scaling group size
        EPS=EPS,
    )
    return y, s


@triton.autotune(configs=quant_kernel_configs, key=["M", "N"])
@triton.jit
def triton_fp8_blockwise_weight_quant_rhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load (block_size x block_size) block of x, where input is row major
    x_offs = offs_m[:, None] * x_stride_dim_0 + offs_n[None, :] * x_stride_dim_1
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Scale the data
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0
    amax = tl.clamp(tl.max(tl.abs(x)), min=EPS, max=float("inf")).to(tl.float64)
    scale = (max_fp8_e4m3 / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=min_fp8_e4m3, max=max_fp8_e4m3).to(y_ptr.dtype.element_ty)

    # Store output in column major format
    y_offs = offs_m[:, None] * y_stride_dim_0 + offs_n[None, :] * y_stride_dim_1
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptr + y_offs, y, mask=y_mask)

    # Write reciprocal scale (scalar value)
    scale_m_off = pid_m * s_stride_dim_0
    scale_n_off = pid_n * s_stride_dim_1
    tl.store(s_ptr + scale_m_off + scale_n_off, tl.div_rn(1.0, scale))


@triton_op("torchao::triton_fp8_blockwise_weight_quant_rhs", mutates_args={})
def triton_fp8_blockwise_weight_quant_rhs(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    assert x.size(0) % block_size == 0 and x.size(1) % block_size == 0, (
        f"Both dimensions of x must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
    ], "dtype must be torch.float8_e4m3fn"
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    y = y.as_strided(y.size(), (1, y.size(0)))  # Column major
    M_blocks, N_blocks = triton.cdiv(M, block_size), triton.cdiv(N, block_size)
    s = x.new_empty(M_blocks, N_blocks, dtype=torch.float32).as_strided(
        (M_blocks, N_blocks),  # shape
        (1, M_blocks),  # stride
    )

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(N, meta["BLOCK_SIZE"]),
        )

    wrap_triton(triton_fp8_blockwise_weight_quant_rhs_kernel)[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M,
        N,
        BLOCK_SIZE=block_size,
        EPS=EPS,
    )
    return y, s


@triton.autotune(configs=quant_kernel_configs, key=["M", "N"])
@triton.jit
def triton_fp8_blockwise_weight_quant_transposed_rhs_kernel(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factors in `s_ptr`.

    Writes output with transposed dims in column-major format.

    Args:
        x_ptr (tl.pointer): Pointer to the input tensor.
        y_ptr (tl.pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (tl.pointer): Pointer to the output tensor where scaling factors will be stored.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Load (block_size x block_size) block of input, where input is row major
    m_offs = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_offs = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs = m_offs[:, None] * x_stride_dim_0 + n_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    # Perform scaling
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0
    amax = tl.clamp(tl.max(tl.abs(x)), min=EPS, max=float("inf")).to(tl.float64)
    scale = (max_fp8_e4m3 / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=min_fp8_e4m3, max=max_fp8_e4m3).to(y_ptr.dtype.element_ty)

    # Write output to column major fomrat
    y_offs = n_offs[:, None] * y_stride_dim_0 + m_offs[None, :] * y_stride_dim_1
    y_mask = (n_offs[:, None] < N) & (m_offs[None, :] < M)
    tl.store(y_ptr + y_offs, y.trans(1, 0), mask=y_mask)

    # Write reciprocal scales
    scale_m = pid_m
    scale_k = pid_n
    scale_offs = scale_k[:, None] * s_stride_dim_0 + scale_m[None, :] * s_stride_dim_1
    scale_mask = (scale_k[:, None] < N // BLOCK_SIZE) & (
        scale_m[None, :] < M // BLOCK_SIZE
    )
    tl.store(s_ptr + scale_offs, tl.div_rn(1.0, scale), mask=scale_mask)


@triton_op("torchao::triton_fp8_blockwise_weight_quant_transposed_rhs", mutates_args={})
def triton_fp8_blockwise_weight_quant_transposed_rhs(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    assert x.size(0) % block_size == 0 and x.size(1) % block_size == 0, (
        f"Both dimensions of x must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
    ], "dtype must be torch.float8_e4m3fn"
    M, N = x.size()
    y = torch.empty(N, M, dtype=dtype, device=x.device)
    y = y.as_strided(y.size(), (1, y.size(0)))  # Column major
    n_blocks, m_blocks = triton.cdiv(N, block_size), triton.cdiv(M, block_size)
    s = x.new_empty(n_blocks, m_blocks, dtype=torch.float32).as_strided(
        (n_blocks, m_blocks),  # shape
        (1, n_blocks),  # stride
    )

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE"]),
            triton.cdiv(N, meta["BLOCK_SIZE"]),
        )

    wrap_triton(triton_fp8_blockwise_weight_quant_transposed_rhs_kernel)[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M,
        N,
        BLOCK_SIZE=block_size,
        EPS=EPS,
    )
    return y, s


def torch_blockwise_scale_act_quant_lhs(x, tile_size=128):
    """
    Input: weight tensor in high precision
    Output: weight tensor in float8, and scale, tiled 1 by tile_size
    """
    assert x.is_contiguous(), "input tensor must be contiguous"
    orig_shape = x.shape

    # Reshape 2D+ input tensor into 2D tensor with shape (leading_dims, tile_size)
    x = x.reshape(-1, tile_size)

    # Compute amax along last dim (i.e., the block)
    x_amax = x.abs().max(dim=1, keepdim=True).values.to(torch.float64)
    x_amax = torch.clamp(x_amax, min=EPS, max=float("inf"))

    # Convert amax to scale
    fp8_dtype_max, fp8_dtype_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )
    s = (fp8_dtype_max / x_amax).to(torch.float32)

    # Apply scale and clamp
    x = (x * s).clamp(min=fp8_dtype_min, max=fp8_dtype_max).to(torch.float8_e4m3fn)

    # Reshape quantized output back to original shape and reshape scales accordingly
    x = x.reshape(*orig_shape)
    s = s.reshape(orig_shape[0], -1).to(torch.float)
    s = s.transpose(-2, -1).contiguous().transpose(-2, -1)

    # Return output tensor and reciprocal scale
    return x, 1.0 / s


def torch_blockwise_scale_act_quant_rhs(
    x: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [torch.float8_e4m3fn], "dtype must be torch.float8_e4m3fn"
    # for debug
    print(f"torch_blockwise_scale_act_quant_rhs: x.shape={x.shape}, block_size={block_size}, dtype={dtype}, eps={eps}")
    M, K = x.size()
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0

    # Reshape input to work with blocks of size (block_size, 1) along dimension 0
    num_blocks_m = M // block_size

    # Reshape to (num_blocks_m, block_size, K) for block processing
    x_blocks = x.view(num_blocks_m, block_size, K)

    # Initialize output tensors
    y_blocks = torch.empty_like(x_blocks, dtype=dtype)
    scales = torch.empty(num_blocks_m, K, dtype=torch.float32, device=x.device)

    # Process each column (K dimension) separately
    for k in range(K):
        # Extract column k from all blocks: shape (num_blocks_m, block_size)
        x_col = x_blocks[:, :, k]  # (num_blocks_m, block_size)

        # Compute absolute max for each block
        amax = torch.abs(x_col).max(dim=1, keepdim=True)[0]  # (num_blocks_m, 1)

        # Clamp to avoid division by zero
        amax = torch.clamp(amax, min=eps).to(torch.float64)

        # Compute scales
        scale = (max_fp8_e4m3 / amax).to(torch.float32)  # (num_blocks_m, 1)

        # Apply scaling
        y_col = x_col * scale  # (num_blocks_m, block_size)

        # Clamp to FP8 range
        y_col = torch.clamp(y_col, min=min_fp8_e4m3, max=max_fp8_e4m3)

        # Store results
        y_blocks[:, :, k] = y_col.to(dtype)
        scales[:, k] = scale.squeeze(-1)  # (num_blocks_m,)

    # Reshape back to original shape (removing padding if any)
    y = y_blocks.view(-1, K)[:M, :]  # (M, K)

    # Convert to column-major format
    y = y.t().contiguous().t()

    # Return output tensor and reciprocal scales
    return y, 1.0 / scales


def torch_blockwise_scale_weight_quant(x, tile_size=128):
    """
    Input: weight tensor in high precision
    Output: weight tensor in float8, and scale, tiled tile_size by tile_size
    """
    assert len(x.shape) == 2, "input shape must be 2D"
    assert x.is_contiguous(), "input tensor must be contiguous"
    height, width = x.shape

    # Compute block sizes
    t_h = height // tile_size
    t_w = width // tile_size

    # Reshape 2D input tensor into 4D tensor with shape (t_h, t_w, tile_size * tile_size)
    x = x.reshape(t_h, tile_size, t_w, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, tile_size * tile_size)

    # Compute amax along last dim (i.e., the block)
    x_amax = x.abs().max(dim=1).values.unsqueeze(1).to(torch.float64)
    x_amax = torch.clamp(x_amax, min=EPS, max=float("inf"))

    # Convert amax to scale
    fp8_dtype_max, fp8_dtype_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )
    s = (fp8_dtype_max / x_amax).to(torch.float32)

    # Apply scale and clamp
    x = (x * s).clamp(min=fp8_dtype_min, max=fp8_dtype_max).to(torch.float8_e4m3fn)

    # Reshape quantized output and scales back to 2D
    x = x.reshape(t_h, t_w, tile_size, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(height, width)
    s = s.reshape(t_h, t_w).to(torch.float)

    # Return output tensor and reciprocal scale
    return x, 1.0 / s
