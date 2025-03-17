#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implementations of the linear cross entropy with token entropy kernel.
"""

import typing
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class EntropyReductionEnum:
    """
    Enum for the reduction method of cross entropy.
    """
    _None = 0
    _Sum = 1
    _Mean = 2


def get_entropy_reduction_enum_number(reduction: str) -> int:
    """
    Get the enum number for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if reduction == "none":
        _enum = EntropyReductionEnum._None
    elif reduction == "sum":
        _enum = EntropyReductionEnum._Sum
    elif reduction == "mean":
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return _enum


def get_entropy_reduction_enum(ce_reduction: int) -> EntropyReductionEnum:
    """
    Get the enum for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if ce_reduction == 0:
        _enum = EntropyReductionEnum._None
    elif ce_reduction == 1:
        _enum = EntropyReductionEnum._Sum
    elif ce_reduction == 2:
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid ce_reduction: {ce_reduction}")
    return _enum


@dataclass
class BackwardEnum:
    """
    Enum for the backward method.
    """
    _Total_Fuse_MN = 0  # Fuse d_logits & d_hidden & d_weight, no intermediate storage, requires fp32 for d_hidden & d_weight
    _Total_Separate = 1  # Store d_logits, no special requirements for d_hidden & d_weight


_BACKWARD: BackwardEnum = BackwardEnum._Total_Separate


def set_backward_method(backward_method: BackwardEnum):
    """
    Set the backward method.
    """
    global _BACKWARD
    _BACKWARD = backward_method


@triton.autotune(
    configs=[triton.Config({
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64
    }, num_stages=3, num_warps=4)],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_kernel_general_mainloop(
        hidden_ptr,
        weight_ptr,
        labels_ptr,
        num_tokens,
        hidden_size,
        vocab_size,
        vocab_per_split,
        stride_hidden_m,
        stride_hidden_k,
        stride_weight_k,
        stride_weight_n,
        max_ptr,
        stride_max_m,
        stride_max_n,
        accu_ptr,
        stride_accu_m,
        stride_accu_n,
        entropy_b_ptr,
        stride_entropy_b_m,
        stride_entropy_b_n,
        global_logprobs_ptr,
        stride_global_logprobs,
        global_logprobs_scalar_ptr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr):
    """
    forward mainloop
    """
    pid = tl.program_id(axis=0)
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if pid_m == 0 and pid_n == 0:
        tl.store(global_logprobs_scalar_ptr, 0.0)

    # create pointers for the first blocks of hidden
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    # load labels for this block
    labels = tl.load(labels_ptr + offs_am, mask=offs_am < num_tokens)

    # traverse over N dimension
    # _max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(0, num_pid_n):
        offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)

        # iterate over K dimension
        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            # load the next block of hidden and weight
            _hidden = tl.load(hidden_ptrs,
                              mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                              other=0.0)
            _weight = tl.load(weight_ptrs,
                              mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < (min(
                                  (pid_n + 1) * vocab_per_split, vocab_size))),
                              other=0.0)

            # GEMM
            logits = tl.dot(_hidden, _weight, logits)

            # advance the ptrs to the next K block
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        # reset hidden_ptrs for next iteration
        hidden_ptrs -= hidden_size * stride_hidden_k

        # update global maximum
        _max_old = _max
        m_pid_n = tl.max(logits, axis=1)
        _max = tl.maximum(_max_old, m_pid_n)

        exp_logits = tl.exp(logits - _max[:, None])
        coeff = tl.exp(_max_old - _max)
        _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

        _entropy_b = _entropy_b * coeff + tl.sum(logits * exp_logits, axis=1)

        label_mask = offs_bn[None, :] == labels[:, None]
        _logprobs += tl.sum(logits * label_mask, axis=1)

    # store maximum
    offs_max_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_max_n = pid_n
    maximum_ptrs = max_ptr + offs_max_n * stride_max_n + offs_max_m * stride_max_m
    tl.store(maximum_ptrs, _max, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))

    # store entropy
    accu_ptrs = accu_ptr + offs_max_n * stride_accu_n + offs_max_m * stride_accu_m
    tl.store(accu_ptrs, _accu, mask=(offs_max_m < num_tokens) & (offs_max_n[None] < num_splits))
    entropy_b_ptrs = entropy_b_ptr + offs_max_n * stride_entropy_b_n + offs_max_m * stride_entropy_b_m
    tl.store(entropy_b_ptrs, _entropy_b, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))

    # store logprobs
    mask = (labels >= pid_n * vocab_per_split) & (labels < min((pid_n + 1) * vocab_per_split, vocab_size))
    mask &= (offs_am < num_tokens)
    global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
    # tl.atomic_add(global_logprobs_ptrs, _logprobs, mask=mask)
    tl.store(global_logprobs_ptrs, _logprobs, mask=mask)


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})], key=["num_tokens", "num_splits"])
@triton.jit
def efficient_entropy_triton_kernel_epilogue(max_ptr, stride_max_m, stride_max_n, num_tokens, num_splits,
                                             global_max_ptr, stride_global_max, accu_ptr, stride_accu_m, stride_accu_n,
                                             global_accu_ptr, stride_global_accu, entropy_b_ptr, stride_entropy_b_m,
                                             stride_entropy_b_n, global_entropy_ptr, stride_global_entropy,
                                             global_logprobs_ptr, stride_global_logprobs, global_logprobs_scalar_ptr,
                                             reduction: int, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    foward epilogue
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        max_ptrs = max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n

        _max = tl.load(max_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0)

        accu_ptrs = accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n
        _accu = tl.load(accu_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0)

        entropy_b_ptrs = entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n
        _entropy_b = tl.load(entropy_b_ptrs,
                             mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                             other=0.0)

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:, None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
        global_entropy_b = _coeff * global_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

    # store
    maximum_ptrs = global_max_ptr + offs_m * stride_global_max
    tl.store(maximum_ptrs, global_max, mask=offs_m < num_tokens)

    # store entropy
    global_accu_ptrs = global_accu_ptr + offs_m * stride_global_accu
    tl.store(global_accu_ptrs, global_accu, mask=offs_m < num_tokens)
    global_entropy_b = tl.fdiv(global_entropy_b, global_accu)  # entropy_b
    global_entropy_b = tl.log(global_accu) + global_max - global_entropy_b  # entropy_a
    global_entropy_ptrs = global_entropy_ptr + offs_m * stride_global_entropy
    tl.store(global_entropy_ptrs, global_entropy_b, mask=offs_m < num_tokens)
    # update logprobs
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    global_logprobs = global_max + tl.log(global_accu) - global_logprobs

    global_logprobs = -1 * global_logprobs
    if reduction == 0:
        tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)
    elif reduction == 1:
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)
    elif reduction == 2:
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0) / num_tokens.to(tl.float32)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)


def efficient_entropy_foward(hidden: torch.Tensor,
                             weight: torch.Tensor,
                             labels: torch.Tensor,
                             reduction: typing.Optional[int] = 2) -> typing.List[torch.Tensor]:
    """
    forward host function
    """
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[0]

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    hidden_size, vocab_size = weight.shape
    assert hidden_size % 128 == 0
    assert vocab_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    if REDUCTION == EntropyReductionEnum._None:
        logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    elif REDUCTION in (EntropyReductionEnum._Sum, EntropyReductionEnum._Mean):
        logprobs = torch.empty((), device=hidden.device, dtype=torch.float32)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    entropy = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    assert logprobs.is_contiguous() and entropy.is_contiguous()

    maximum = torch.empty_like(entropy)
    acc = torch.empty_like(entropy)
    assert maximum.is_contiguous() and acc.is_contiguous()

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _entropy_b = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)

    if REDUCTION == EntropyReductionEnum._None:
        _logprobs = logprobs
    else:
        _logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)

    assert _accu.is_contiguous() and _entropy_b.is_contiguous() and _max.is_contiguous()
    assert _accu.is_cuda and _entropy_b.is_cuda and _max.is_cuda

    # 1D kernel launch, then split the tile
    def mainloop_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * num_splits,)

    efficient_entropy_kernel_general_mainloop[mainloop_grid](hidden, weight, labels, num_tokens, hidden_size,
                                                             vocab_size, vocab_per_split, hidden.stride(0),
                                                             hidden.stride(1), weight.stride(0), weight.stride(1), _max,
                                                             _max.stride(0), _max.stride(1), _accu, _accu.stride(0),
                                                             _accu.stride(1), _entropy_b, _entropy_b.stride(0),
                                                             _entropy_b.stride(1), _logprobs, _logprobs.stride(0),
                                                             logprobs)

    # reduction on maximum and maximum_indices
    def epilogue_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

    efficient_entropy_triton_kernel_epilogue[epilogue_grid](_max, _max.stride(0), _max.stride(1), num_tokens,
                                                            num_splits, maximum, maximum.stride(0), _accu,
                                                            _accu.stride(0), _accu.stride(1), acc,
                                                            acc.stride(0), _entropy_b, _entropy_b.stride(0),
                                                            _entropy_b.stride(1), entropy, entropy.stride(0), _logprobs,
                                                            _logprobs.stride(0), logprobs, REDUCTION)

    return (logprobs, entropy, maximum, acc)


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64
        }, num_stages=3, num_warps=4),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_preprocess(num_tokens: int, hidden_size: int, vocab_size: int,
                                                         vocab_per_split: int, num_splits: int, hidden_ptr,
                                                         stride_hidden_m, stride_hidden_k, weight_ptr, stride_weight_k,
                                                         stride_weight_n, maximum_ptr, d_scale_ptr, stride_d_scale_m,
                                                         stride_d_scale_n, BLOCK_SIZE_M: tl.constexpr,
                                                         BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    backward preprocess
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # pointers for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    maximum = tl.load(maximum_ptr + offs_am, mask=offs_am < num_tokens)

    _scale = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(0, num_pid_n):
        offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            _hidden = tl.load(hidden_ptrs,
                              mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                              other=0.0)
            _weight = tl.load(weight_ptrs,
                              mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < min(
                                  (pid_n + 1) * vocab_per_split, vocab_size)),
                              other=0.0)
            logits = tl.dot(_hidden, _weight, logits)

            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        hidden_ptrs -= hidden_size * stride_hidden_k

        exp_logits = tl.exp(logits - maximum[:, None])

        _scale += tl.sum(exp_logits * logits, axis=1)

    # tl.store(d_scale_ptr + offs_am * stride_d_scale_m + pid_n * stride_d_scale_n,
    #          _scale, mask=offs_am < num_tokens)
    tl.store(d_scale_ptr + pid_n * stride_d_scale_m + offs_am * stride_d_scale_n,
             _scale,
             mask=(offs_am < num_tokens) & (pid_n < num_splits))


# NOTE: merge d_weight & d_hidden here, split along M & N
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 16
        },
                      num_stages=3,
                      num_warps=4)
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_mainloop_MN(
        num_tokens: int, hidden_size: int, vocab_size: int, hidden_ptr, stride_hidden_m, stride_hidden_k, weight_ptr,
        stride_weight_k, stride_weight_n, labels_ptr, stride_labels, maximum_ptr, stride_maximum, accu_ptr, stride_accu,
        d_entropy_ptr, stride_d_entropy, d_logprobs_ptr, stride_d_logprobs, reduction: int, d_scale_ptr, stride_d_scale,
        d_hidden_ptr, stride_d_hidden_m, stride_d_hidden_k, d_weight_ptr, stride_d_weight_k, stride_d_weight_n,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """
    backward mainloop, where d_logits & d_hidden & d_weight are fused
    """
    # block swizzling
    # pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum_ptrs = maximum_ptr + offs_am * stride_maximum
    maximum = tl.load(maximum_ptrs, mask=offs_am < num_tokens, other=0.0)
    accu_ptrs = accu_ptr + offs_am * stride_accu
    accu = tl.load(accu_ptrs, mask=offs_am < num_tokens, other=1e-6)  # epsilon to avoid division by zero
    accu_rcp = tl.fdiv(1.0, accu)

    d_entropy_ptrs = d_entropy_ptr + offs_am * stride_d_entropy
    d_entropy = tl.load(d_entropy_ptrs, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:  # none
        d_logprobs_ptrs = d_logprobs_ptr + offs_am * stride_d_logprobs
        d_logprobs = tl.load(d_logprobs_ptrs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:  # sum
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:  # mean
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    d_scale = tl.load(d_scale_ptr + offs_am * stride_d_scale, mask=offs_am < num_tokens, other=0.0)

    hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
    labels_ptrs = labels_ptr + offs_am * stride_labels
    labels = tl.load(labels_ptrs, mask=offs_am < num_tokens, other=0)

    d_hidden_ptrs = d_hidden_ptr + offs_am[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
    d_weight_ptrs = d_weight_ptr + offs_k[:, None] * stride_d_weight_k + offs_bn[None, :] * stride_d_weight_n

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        _hidden = tl.load(hidden_ptrs,
                          mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                          other=0.0)
        _weight = tl.load(weight_ptrs,
                          mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < vocab_size),
                          other=0.0)

        logits = tl.dot(_hidden, _weight, logits)

        hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k
    hidden_ptrs -= hidden_size * stride_hidden_k
    weight_ptrs -= hidden_size * stride_weight_k

    exp_logits = tl.exp(logits - maximum[:, None])

    d_pd = logits * -d_entropy[:, None]
    mask = offs_bn[None, :] == labels[:, None]
    d_pd += tl.fdiv((-1.0 * d_logprobs * accu)[:, None], exp_logits) * mask

    coeff = d_scale * d_entropy * accu_rcp * accu_rcp + d_logprobs * accu_rcp
    d_logits = exp_logits * coeff[:, None]
    d_logits += exp_logits * d_pd * accu_rcp[:, None]

    # loop for d_weight & d_hidden
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        _hidden = tl.load(hidden_ptrs,
                          mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                          other=0.0)
        _d_weight = tl.dot(tl.trans(_hidden).to(tl.float32), d_logits)
        tl.atomic_add(d_weight_ptrs,
                      _d_weight,
                      mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < vocab_size))

        _weight = tl.load(weight_ptrs,
                          mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < vocab_size),
                          other=0.0)
        _d_hidden = tl.dot(d_logits, tl.trans(_weight).to(tl.float32))
        tl.atomic_add(d_hidden_ptrs,
                      _d_hidden,
                      mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens))

        hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        d_hidden_ptrs += BLOCK_SIZE_K * stride_d_hidden_k
        d_weight_ptrs += BLOCK_SIZE_K * stride_d_weight_k


# NOTE: split tile from d_logits' perspective
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 16
        },
                      num_stages=3,
                      num_warps=4),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits(
        num_tokens: int, hidden_size: int, vocab_size: int, hidden_ptr, stride_hidden_m, stride_hidden_k, weight_ptr,
        stride_weight_k, stride_weight_n, labels_ptr, stride_labels, maximum_ptr, stride_maximum, accu_ptr, stride_accu,
        d_entropy_ptr, stride_d_entropy, d_logprobs_ptr, stride_d_logprobs, reduction: int, d_scale_ptr, stride_d_scale,
        d_logits_ptr, stride_d_logits_m, stride_d_logits_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """
    backward d_logits
    """
    # block swizzling
    # pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum_ptrs = maximum_ptr + offs_am * stride_maximum
    maximum = tl.load(maximum_ptrs, mask=offs_am < num_tokens, other=0.0)
    accu_ptrs = accu_ptr + offs_am * stride_accu
    accu = tl.load(accu_ptrs, mask=offs_am < num_tokens, other=1e-6)  # epsilon to avoid division by zero
    accu_rcp = tl.fdiv(1.0, accu)

    d_entropy_ptrs = d_entropy_ptr + offs_am * stride_d_entropy
    d_entropy = tl.load(d_entropy_ptrs, mask=offs_am < num_tokens, other=0.0)
    if reduction == 0:  # none
        d_logprobs_ptrs = d_logprobs_ptr + offs_am * stride_d_logprobs
        d_logprobs = tl.load(d_logprobs_ptrs, mask=offs_am < num_tokens, other=0.0)
    elif reduction == 1:  # sum
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:  # mean
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    d_scale = tl.load(d_scale_ptr + offs_am * stride_d_scale, mask=offs_am < num_tokens, other=0.0)

    # d_acc_exp_logits = d_scale * d_entropy * accu_rcp * accu_rcp
    # d_acc_exp_logits += d_logprobs * accu_rcp
    # d_acc_exp_logits += d_entropy * accu_rcp

    # These equal to d_max = d_entropy
    # d_max = d_scale * -d_entropy * accu_rcp
    # d_max -= d_logprobs
    # d_max += accu * d_acc_exp_logits

    hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
    labels_ptrs = labels_ptr + offs_am * stride_labels
    labels = tl.load(labels_ptrs, mask=offs_am < num_tokens, other=0)

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        _hidden = tl.load(hidden_ptrs,
                          mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                          other=0.0)
        _weight = tl.load(weight_ptrs,
                          mask=(offs_k[:, None] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[None, :] < vocab_size),
                          other=0.0)

        logits = tl.dot(_hidden, _weight, logits)

        hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k
    hidden_ptrs -= hidden_size * stride_hidden_k
    weight_ptrs -= hidden_size * stride_weight_k

    exp_logits = tl.exp(logits - maximum[:, None])

    d_pd = logits * -d_entropy[:, None]
    mask = offs_bn[None, :] == labels[:, None]
    d_pd += tl.fdiv((-1.0 * d_logprobs * accu)[:, None], exp_logits) * mask

    coeff = d_scale * d_entropy * accu_rcp * accu_rcp + d_logprobs * accu_rcp
    d_logits = exp_logits * coeff[:, None]
    d_logits += exp_logits * d_pd * accu_rcp[:, None]
    # d_logits += exp_logits * logits * (-d_entropy * accu_rcp)[:,None]
    # d_logits -= tl.where(mask, d_logprobs[:,None], 0.0)

    # d_max is always zeros
    # d_max = d_entropy - d_max
    # mask = offs_bn[None,:] == maximum_indices[:,None]
    # d_logits += tl.where(mask, d_max[:,None], 0.0)

    # store d_logits
    d_logits_ptrs = d_logits_ptr + offs_am[:, None] * stride_d_logits_m + offs_bn[None, :] * stride_d_logits_n
    tl.store(d_logits_ptrs,
             d_logits.to(hidden_ptr.dtype.element_ty),
             mask=(offs_am[:, None] < num_tokens) & (offs_bn[None, :] < vocab_size))


def efficient_entropy_backward(dlogprobs: torch.Tensor,
                               dentropy: torch.Tensor,
                               hidden: torch.Tensor,
                               weight: torch.Tensor,
                               labels: torch.Tensor,
                               maximum: torch.Tensor,
                               acc: torch.Tensor,
                               reduction: typing.Optional[int] = 2) -> typing.List[torch.Tensor]:
    """
    backward host function
    """
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[0]

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    hidden_size, vocab_size = weight.shape
    assert hidden_size % 128 == 0
    assert vocab_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    if REDUCTION == EntropyReductionEnum._None:
        assert dlogprobs.shape == (num_tokens,)
    else:
        assert dlogprobs.dim() == 0

    assert dlogprobs.is_contiguous() and dentropy.is_contiguous()
    assert dlogprobs.is_cuda and dentropy.is_cuda
    assert dlogprobs.device == hidden.device and dlogprobs.device == dentropy.device
    assert dentropy.shape == (num_tokens,)

    d_hidden, d_weight = None, None
    if _BACKWARD == BackwardEnum._Total_Fuse_MN:
        d_hidden = torch.zeros_like(hidden, dtype=torch.float32, device=hidden.device)
        d_weight = torch.zeros_like(weight, dtype=torch.float32, device=weight.device)
    elif _BACKWARD == BackwardEnum._Total_Separate:
        d_hidden = torch.empty_like(hidden, dtype=hidden.dtype, device=hidden.device)
        d_weight = torch.empty_like(weight, dtype=hidden.dtype, device=weight.device)
    assert d_hidden.is_contiguous() and d_weight.is_contiguous()

    assert maximum.is_contiguous() and acc.is_contiguous()
    assert maximum.device == hidden.device and acc.device == hidden.device
    assert maximum.shape == labels.shape == acc.shape
    assert maximum.is_cuda and acc.is_cuda

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    # _d_scale_non_reduced = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _d_scale_non_reduced = torch.empty((num_splits, num_tokens), device=hidden.device, dtype=torch.float32)
    assert _d_scale_non_reduced.is_contiguous() and _d_scale_non_reduced.is_cuda

    def preprocess_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * num_splits,)

    efficient_entropy_backward_kernel_general_preprocess[preprocess_grid](
        num_tokens,
        hidden_size,
        vocab_size,
        vocab_per_split,
        num_splits,
        hidden,
        hidden.stride(0),
        hidden.stride(1),
        weight,
        weight.stride(0),
        weight.stride(1),
        maximum,
        _d_scale_non_reduced,
        _d_scale_non_reduced.stride(0),
        _d_scale_non_reduced.stride(1),
    )
    _d_scale = _d_scale_non_reduced.sum(dim=0)

    if _BACKWARD == BackwardEnum._Total_Fuse_MN:

        def mainloop_grid(meta):
            return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * triton.cdiv(vocab_size, meta["BLOCK_SIZE_N"]),)

        efficient_entropy_backward_kernel_general_mainloop_MN[mainloop_grid](
            num_tokens,
            hidden_size,
            vocab_size,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs,
            dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
            REDUCTION,
            _d_scale,
            _d_scale.stride(0),
            d_hidden,
            d_hidden.stride(0),
            d_hidden.stride(1),
            d_weight,
            d_weight.stride(0),
            d_weight.stride(1),
        )
    elif _BACKWARD == BackwardEnum._Total_Separate:
        _d_logits = torch.empty((num_tokens, vocab_size), device=hidden.device, dtype=hidden.dtype)

        def d_logits_grid(meta):
            return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * triton.cdiv(vocab_size, meta["BLOCK_SIZE_N"]),)

        efficient_entropy_backward_kernel_general_d_logits[d_logits_grid](
            num_tokens,
            hidden_size,
            vocab_size,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs,
            dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
            REDUCTION,
            _d_scale,
            _d_scale.stride(0),
            _d_logits,
            _d_logits.stride(0),
            _d_logits.stride(1),
        )

        torch.matmul(_d_logits, weight.T, out=d_hidden)
        torch.matmul(hidden.T, _d_logits, out=d_weight)
    return d_hidden, d_weight
