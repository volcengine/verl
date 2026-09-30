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

import typing

import torch
import torch.distributed as dist

_LIGER_TP_BOOTSTRAP_GLOBAL_RANKS: tuple[int, ...] | None = None
_LIGER_TP_NATIVE_FUNCTION = None
_LIGER_TP_NVSHMEM = None


def _validate_liger_tp_device() -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("The Liger TP-FLSCE backend requires an NVIDIA CUDA GPU")

    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability != (9, 0) and capability[0] != 10:
        raise RuntimeError(
            "The Liger TP-FLSCE backend supports Hopper (SM90) and Blackwell (SM10x) GPUs, "
            f"but the current device has compute capability {capability[0]}.{capability[1]}"
        )


def _require_liger_tp_runtime():
    global _LIGER_TP_NATIVE_FUNCTION, _LIGER_TP_NVSHMEM

    if _LIGER_TP_NATIVE_FUNCTION is not None and _LIGER_TP_NVSHMEM is not None:
        return _LIGER_TP_NATIVE_FUNCTION, _LIGER_TP_NVSHMEM

    try:
        from liger_cute_kernels import nvshmem
        from liger_kernel.ops.cute.fused_linear_scaled_cross_entropy_tp import (
            LigerFusedLinearScaledCrossEntropyNativeTPFunction,
            is_available,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The Liger TP-FLSCE backend requires `liger-kernel[lck]>=0.8.3` and a loadable "
            "`liger-cute-kernels` native wheel"
        ) from exc

    try:
        native_available = is_available()
    except (ImportError, OSError) as exc:
        raise RuntimeError("The Liger TP-FLSCE native libraries could not be loaded") from exc

    if not native_available:
        raise RuntimeError(
            "The Liger TP-FLSCE native runtime is unavailable. Install `liger-kernel[lck]>=0.8.3` "
            "with the LCK wheel matching this CUDA environment."
        )

    _LIGER_TP_NATIVE_FUNCTION = LigerFusedLinearScaledCrossEntropyNativeTPFunction
    _LIGER_TP_NVSHMEM = nvshmem
    return _LIGER_TP_NATIVE_FUNCTION, _LIGER_TP_NVSHMEM


def _process_group_global_ranks(process_group: dist.ProcessGroup) -> tuple[int, ...]:
    return tuple(
        dist.get_global_rank(process_group, group_rank) for group_rank in range(dist.get_world_size(process_group))
    )


def initialize_liger_tp_flsce(process_group: dist.ProcessGroup) -> None:
    """Initialize one NVSHMEM world matching the calling rank's Megatron TP group."""
    global _LIGER_TP_BOOTSTRAP_GLOBAL_RANKS

    if process_group is None:
        raise ValueError("A tensor-parallel process group is required for the Liger TP-FLSCE backend")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before the Liger TP-FLSCE backend")

    _validate_liger_tp_device()
    _, nvshmem = _require_liger_tp_runtime()
    global_ranks = _process_group_global_ranks(process_group)

    if _LIGER_TP_BOOTSTRAP_GLOBAL_RANKS is None:
        nvshmem.init_from_pg(process_group)
        _LIGER_TP_BOOTSTRAP_GLOBAL_RANKS = global_ranks
    elif _LIGER_TP_BOOTSTRAP_GLOBAL_RANKS != global_ranks:
        raise RuntimeError(
            "The Liger TP-FLSCE NVSHMEM runtime was already initialized for tensor-parallel ranks "
            f"{_LIGER_TP_BOOTSTRAP_GLOBAL_RANKS}, but this engine uses {global_ranks}"
        )

    nvshmem.resolve_team(process_group)


def _linear_cross_entropy_liger_tp(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: str,
    dist_process_group: dist.ProcessGroup,
    chunk_size: int,
    tiles_per_reduce: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(reduction, str):
        raise TypeError(f"reduction must be a string, got {type(reduction)}")
    if reduction.lower() != "none":
        raise NotImplementedError("The Liger TP-FLSCE backend currently supports reduction='none' only")
    if dist_process_group is None:
        raise ValueError("A tensor-parallel process group is required for the Liger TP-FLSCE backend")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    if not isinstance(tiles_per_reduce, int) or isinstance(tiles_per_reduce, bool) or tiles_per_reduce not in (1, 2, 4):
        raise ValueError(f"tiles_per_reduce must be one of 1, 2, or 4, got {tiles_per_reduce!r}")
    if hidden.ndim not in (2, 3):
        raise ValueError(f"hidden must be 2D or 3D, got shape {tuple(hidden.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")

    _validate_liger_tp_device()
    native_function, _ = _require_liger_tp_runtime()

    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1).to(torch.int64)
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError(f"hidden has {hidden.shape[0]} tokens, but labels has {labels.shape[0]} elements")
    if hidden.shape[0] == 0:
        raise ValueError("The Liger TP-FLSCE backend requires at least one token")

    vocab_start = dist.get_rank(dist_process_group) * weight.shape[0]
    logprob_chunks = []
    entropy_chunks = []

    for chunk_start in range(0, hidden.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, hidden.shape[0])
        valid_tokens = chunk_end - chunk_start
        hidden_chunk = hidden[chunk_start:chunk_end]
        label_chunk = labels[chunk_start:chunk_end]

        # Always launch the configured chunk width so a short first microbatch
        # cannot permanently fix the native workspace to a smaller capacity.
        if valid_tokens < chunk_size:
            padding = chunk_size - valid_tokens
            hidden_chunk = torch.cat(
                (hidden_chunk, hidden_chunk.new_zeros((padding, hidden_chunk.shape[1]))),
                dim=0,
            )
            label_chunk = torch.cat(
                (label_chunk, label_chunk.new_full((padding,), -100)),
                dim=0,
            )

        nll, entropy = native_function.apply(
            hidden_chunk,
            weight,
            label_chunk,
            vocab_start,
            temperature,
            -100,
            tiles_per_reduce,
            True,
            dist_process_group,
        )
        logprob_chunks.append(-nll[:valid_tokens])
        entropy_chunks.append(entropy[:valid_tokens])

    return torch.cat(logprob_chunks), torch.cat(entropy_chunks)


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: typing.Optional[float] = 1.0,
        reduction: typing.Optional[str] = "none",
        dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    ) -> list[torch.Tensor]:
        """_summary_

        Args:
            ctx (_type_): _description_
            hidden (torch.Tensor): (batch_size, num_tokens, hidden_size) -> (batch_size * num_tokens, hidden_size)
            weight (torch.Tensor): (vocab_size, hidden_size)
            labels (torch.Tensor): (batch_size, num_tokens) -> (batch_size * num_tokens, )
            temperature (typing.Optional[float], optional): _description_. Defaults to 1.0.
            reduction (typing.Optional[str], optional): _description_. Defaults to "none".
            dist_process_group (typing.Optional[dist.ProcessGroup], optional): _description_. Defaults to None.

        Returns:
            typing.List[torch.Tensor]: _description_
        """

        assert isinstance(temperature, float), f"temperature must be a float, but got {type(temperature)}"
        assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            from . import kernels

            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

            original_hidden_shape = hidden.shape
            if len(hidden.shape) != 2:
                hidden = hidden.view(-1, hidden.shape[-1])  # (batch_size * num_tokens, hidden_size)
            if len(labels.shape) != 1:
                labels = labels.view(-1)

            logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(
                hidden, weight, labels, REDUCTION, temperature, dist_process_group
            )

            ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
            ctx.original_hidden_shape = original_hidden_shape
            ctx.REDUCTION = REDUCTION
            ctx.dist_process_group = dist_process_group
            ctx.should_return_fp32_grad = False
            ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> list[torch.Tensor]:
        from . import kernels

        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION
            dist_process_group = ctx.dist_process_group
            should_return_fp32_grad = ctx.should_return_fp32_grad
            temperature = ctx.temperature

            d_hidden, d_weight = kernels.efficient_entropy_backward(
                dlogprobs,
                dentropy,
                hidden,
                weight,
                labels,
                _maximum,
                _accumulate,
                _entropy_b,
                REDUCTION,
                should_return_fp32_grad,
                temperature,
                dist_process_group,
            )
            d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return (d_hidden, d_weight, None, None, None, None)


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    *,
    impl_backend: str = "triton",
    chunk_size: int = 512,
    tiles_per_reduce: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused linear + cross-entropy (+ token entropy).

    Always returns ``(logprobs, entropy)``. On Ascend NPU, dispatches to CANN
    vocab-parallel fused CE when available (``VERL_NPU_LCE_BACKEND=auto|cann``).
    Hybrid backward: CANN CE when ``dentropy==0``; chunked PyTorch CE+entropy
    (Triton-aligned) when ``dentropy!=0``.
    """
    if not isinstance(impl_backend, str):
        raise TypeError(f"impl_backend must be a string, got {type(impl_backend)}")
    impl_backend = impl_backend.lower()
    if temperature is None:
        temperature = 1.0
    if reduction is None:
        reduction = "none"
    if impl_backend in ("torch", "triton"):
        if hidden.device.type == "npu":
            from verl.utils.kernel.npu.cann_linear_ce import CannLinearCrossEntropy, should_use_cann_linear_ce

            if should_use_cann_linear_ce(hidden.device):
                hidden_2d = hidden.view(-1, hidden.shape[-1]) if len(hidden.shape) != 2 else hidden
                labels_1d = labels.view(-1) if len(labels.shape) != 1 else labels
                return CannLinearCrossEntropy.apply(
                    hidden_2d, weight, labels_1d, float(temperature), str(reduction), dist_process_group
                )
            raise RuntimeError(
                "Fused linear_cross_entropy on NPU requires CANN torch_npu APIs "
                "(VERL_NPU_LCE_BACKEND=auto|cann). Disable use_fused_kernels, or install "
                "a torch_npu build that provides the fused linear-CE ops."
            )

        return LinearCrossEntropy.apply(
            hidden, weight, labels, float(temperature), str(reduction), dist_process_group
        )
    if impl_backend in ("liger", "liger_tp"):
        return _linear_cross_entropy_liger_tp(
            hidden,
            weight,
            labels,
            temperature,
            reduction,
            dist_process_group,
            chunk_size,
            tiles_per_reduce,
        )
    raise ValueError(f"Unsupported linear cross entropy backend {impl_backend!r}; choose 'triton' or 'liger_tp'")
