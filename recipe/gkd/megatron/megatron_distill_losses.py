# megatron_distill_losses.py
# A unified file that provides 4 selectable vocab-parallel distillation losses:
#   1) KL      : KL(P_topk || Q_full)  (teacher top-k truncated forward KL)
#   2) RKL     : KL(Q_hat_topk || P_hat_topk)  (pure reverse KL on renormalized top-k)
#   3) KL_RKL  : (1-r)*KL + r*RKL
#   4) JSD     : JSD_beta(P_topk, Q_full) with analytic rest term for Q||M outside top-k
#
# Usage:
#   op = build_vocab_parallel_distill_loss(cfg).cuda()
#   loss_per_token = op(vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices)

import math
from typing import Any, Dict, Optional

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_data_parallel_rank,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.fusions.fused_cross_entropy import calculate_logits_max


# -----------------------------
# utils
# -----------------------------
def _clamp01_open(x: float, eps: float = 1e-6) -> float:
    # clamp into (0, 1) open interval for logs
    if x < eps:
        return eps
    if x > 1.0 - eps:
        return 1.0 - eps
    return x


def mylog(message: str, filename: str = "distill_loss.log"):
    # optional debug
    with open(filename, "a") as f:
        f.write(f"({get_data_parallel_rank()}, {get_tensor_model_parallel_rank()}): {message}\n")


# ============================================================
# 1) Forward KL (teacher top-k truncated): KL(P_topk || Q_full)
# ============================================================
class _VocabParallelKLDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices):
        eps = 1e-20

        # Student Q_full = softmax(logits) (vocab-parallel)
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)

        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        vocab_parallel_logits.exp_()
        exp_logits = vocab_parallel_logits
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        Q_full = exp_logits
        Q_full.div_(sum_exp_logits.unsqueeze(-1))  # [*, V_part]

        # Local vocab range and map global top-k -> local indices
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        topk_in_vocab = (target_topk_indices >= vocab_start) & (target_topk_indices < vocab_end)

        topk_idx_local = (target_topk_indices - vocab_start).clone()
        topk_idx_local[~topk_in_vocab] = 0  # placeholder

        # Teacher P_topk (local pieces only)
        P_topk_part = torch.exp(target_topk_logps).clone()
        P_topk_part[~topk_in_vocab] = 0.0

        logP_topk_part = target_topk_logps.clone()
        logP_topk_part[~topk_in_vocab] = 0.0

        # Gather student's Q on teacher top-k indices
        origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)

        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view(origin_shape).clone()
        Q_topk[~topk_in_vocab] = 0.0

        logQ_topk = torch.log(Q_topk + eps)
        logQ_topk[~topk_in_vocab] = 0.0

        # KL(P_topk || Q_full) ≈ sum_k P_k (logP_k - logQ_k)
        per_token_kl_local = torch.sum(P_topk_part * (logP_topk_part - logQ_topk), dim=-1)  # [*]
        per_token_kl = per_token_kl_local.clone()
        torch.distributed.all_reduce(
            per_token_kl, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        ctx.save_for_backward(Q_full, P_topk_part, topk_idx_local)
        return per_token_kl

    @staticmethod
    def backward(ctx, grad_output):
        Q_full, P_topk_part, topk_idx_local = ctx.saved_tensors
        partition_vocab_size = Q_full.size(-1)
        topk = topk_idx_local.size(-1)

        # d/dz KL(P||Q) = Q - P_sparse(topk)
        grad_input = Q_full.clone()
        grad_2d = grad_input.view(-1, partition_vocab_size)
        row = torch.arange(grad_2d.size(0), device=grad_2d.device).unsqueeze(-1)
        idx_2d = topk_idx_local.view(-1, topk)
        grad_2d[row, idx_2d] -= P_topk_part.view(-1, topk)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None


def vocab_parallel_kl_divergence(vocab_parallel_logits, target_topk_logps, target_topk_indices):
    return _VocabParallelKLDivergence.apply(vocab_parallel_logits, target_topk_logps, target_topk_indices)


# ============================================================
# 2) Pure Reverse KL on top-k (renormalized): KL(Q_hat || P_hat)
# ============================================================
class _VocabParallelRKLDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices):
        eps = 1e-20

        # Student Q_full
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)

        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        vocab_parallel_logits.exp_()
        exp_logits = vocab_parallel_logits
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        Q_full = exp_logits
        Q_full.div_(sum_exp_logits.unsqueeze(-1))  # [*, V_part]

        # Local vocab range + local indices
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        topk_in_vocab = (target_topk_indices >= vocab_start) & (target_topk_indices < vocab_end)

        topk_idx_local = (target_topk_indices - vocab_start).clone()
        topk_idx_local[~topk_in_vocab] = 0  # placeholder

        # Teacher P_topk (local pieces only)
        P_topk_part = torch.exp(target_topk_logps).clone()
        P_topk_part[~topk_in_vocab] = 0.0

        # Gather Q_topk
        origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)

        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view(origin_shape).clone()
        Q_topk[~topk_in_vocab] = 0.0

        # Global sums for renorm
        P_sum_local = P_topk_part.sum(dim=-1)
        P_sum = P_sum_local.clone()
        torch.distributed.all_reduce(P_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        Q_sum_local = Q_topk.sum(dim=-1)
        Q_sum = Q_sum_local.clone()
        torch.distributed.all_reduce(Q_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        Q_hat = Q_topk / (Q_sum.unsqueeze(-1) + eps)
        P_hat = P_topk_part / (P_sum.unsqueeze(-1) + eps)

        logQ_hat = torch.log(Q_hat + eps)
        logP_hat = torch.log(P_hat + eps)

        per_token_rkl_local = torch.sum(Q_hat * (logQ_hat - logP_hat), dim=-1)
        per_token_rkl = per_token_rkl_local.clone()
        torch.distributed.all_reduce(
            per_token_rkl, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        ctx.save_for_backward(Q_full, P_topk_part, topk_idx_local, Q_sum, P_sum)
        return per_token_rkl

    @staticmethod
    def backward(ctx, grad_output):
        eps = 1e-20
        Q_full, P_topk_part, topk_idx_local, Q_sum, P_sum = ctx.saved_tensors

        partition_vocab_size = Q_full.size(-1)
        topk = topk_idx_local.size(-1)

        # Re-gather Q_topk
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row1 = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row1.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view_as(P_topk_part)

        # Only real local entries
        topk_mask = P_topk_part > 0
        Q_topk = torch.where(topk_mask, Q_topk, torch.zeros_like(Q_topk))

        Z = Q_sum.unsqueeze(-1) + eps
        T = P_sum.unsqueeze(-1) + eps

        Q_hat = Q_topk / Z
        P_hat = P_topk_part / T
        logQ_hat = torch.log(Q_hat + eps)
        logP_hat = torch.log(P_hat + eps)

        a = logQ_hat + 1.0 - logP_hat

        # mean_a = sum(Q_hat * a) over global topk
        mean_a_local = torch.sum(Q_hat * a, dim=-1)
        mean_a = mean_a_local.clone()
        torch.distributed.all_reduce(mean_a, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        # grad on topk then scatter into vocab partition
        grad_topk = (Q_topk / Z) * (a - mean_a.unsqueeze(-1))  # [*, K]

        grad_input = torch.zeros_like(Q_full)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        grad_topk_2d = grad_topk.view(-1, topk)
        grad_topk_2d = torch.where(topk_mask.view(-1, topk), grad_topk_2d, torch.zeros_like(grad_topk_2d))

        idx_2d = topk_idx_local.view(-1, topk)
        row2 = torch.arange(grad_2d.size(0), device=grad_2d.device).unsqueeze(-1)
        grad_2d[row2, idx_2d] += grad_topk_2d

        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None


def vocab_parallel_rkl_divergence(vocab_parallel_logits, target_topk_logps, target_topk_indices):
    return _VocabParallelRKLDivergence.apply(vocab_parallel_logits, target_topk_logps, target_topk_indices)


# ============================================================
# 3) KL + RKL weighted: (1-r)*KL + r*RKL
# ============================================================
class _VocabParallelWeightedKLRKLDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices, rkl_ratio: float = 0.1):
        eps = 1e-20
        rkl_ratio = float(rkl_ratio)
        if rkl_ratio < 0.0:
            rkl_ratio = 0.0
        if rkl_ratio > 1.0:
            rkl_ratio = 1.0
        kl_ratio = 1.0 - rkl_ratio

        # Student Q_full
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)

        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        vocab_parallel_logits.exp_()
        exp_logits = vocab_parallel_logits
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        Q_full = exp_logits
        Q_full.div_(sum_exp_logits.unsqueeze(-1))  # [*, V_part]

        # Local vocab range + local indices
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        topk_in_vocab = (target_topk_indices >= vocab_start) & (target_topk_indices < vocab_end)

        topk_idx_local = (target_topk_indices - vocab_start).clone()
        topk_idx_local[~topk_in_vocab] = 0

        # Teacher P_topk (local)
        P_topk_part = torch.exp(target_topk_logps).clone()
        P_topk_part[~topk_in_vocab] = 0.0

        logP_topk_part = target_topk_logps.clone()
        logP_topk_part[~topk_in_vocab] = 0.0

        # Gather Q_topk
        origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)

        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view(origin_shape).clone()
        Q_topk[~topk_in_vocab] = 0.0

        logQ_topk = torch.log(Q_topk + eps)
        logQ_topk[~topk_in_vocab] = 0.0

        # Forward KL (not renorm): sum P_k (logP_k - logQ_k)
        per_token_kl_local = torch.sum(P_topk_part * (logP_topk_part - logQ_topk), dim=-1)
        per_token_kl = per_token_kl_local.clone()
        torch.distributed.all_reduce(per_token_kl, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        # Reverse KL on topk with renorm
        P_sum_local = P_topk_part.sum(dim=-1)
        P_sum = P_sum_local.clone()
        torch.distributed.all_reduce(P_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        Q_sum_local = Q_topk.sum(dim=-1)
        Q_sum = Q_sum_local.clone()
        torch.distributed.all_reduce(Q_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        Q_hat = Q_topk / (Q_sum.unsqueeze(-1) + eps)
        P_hat = P_topk_part / (P_sum.unsqueeze(-1) + eps)
        logQ_hat = torch.log(Q_hat + eps)
        logP_hat = torch.log(P_hat + eps)

        per_token_rkl_local = torch.sum(Q_hat * (logQ_hat - logP_hat), dim=-1)
        per_token_rkl = per_token_rkl_local.clone()
        torch.distributed.all_reduce(per_token_rkl, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        per_token_loss = kl_ratio * per_token_kl + rkl_ratio * per_token_rkl

        ctx.save_for_backward(Q_full, P_topk_part, topk_idx_local, Q_sum, P_sum)
        ctx.rkl_ratio = rkl_ratio
        ctx.kl_ratio = kl_ratio
        return per_token_loss

    @staticmethod
    def backward(ctx, grad_output):
        eps = 1e-20
        Q_full, P_topk_part, topk_idx_local, Q_sum, P_sum = ctx.saved_tensors
        rkl_ratio = ctx.rkl_ratio
        kl_ratio = ctx.kl_ratio

        partition_vocab_size = Q_full.size(-1)
        topk = topk_idx_local.size(-1)

        # A) Forward KL grad = Q_full - P_sparse(topk)
        grad_kl = Q_full.clone()
        grad_kl_2d = grad_kl.view(-1, partition_vocab_size)
        row = torch.arange(grad_kl_2d.size(0), device=grad_kl_2d.device).unsqueeze(-1)
        idx_2d = topk_idx_local.view(-1, topk)
        grad_kl_2d[row, idx_2d] -= P_topk_part.view(-1, topk)

        # B) Reverse KL grad (scatter topk only)
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row1 = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row1.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view_as(P_topk_part)

        topk_mask = P_topk_part > 0
        Q_topk = torch.where(topk_mask, Q_topk, torch.zeros_like(Q_topk))

        Z = Q_sum.unsqueeze(-1) + eps
        T = P_sum.unsqueeze(-1) + eps

        Q_hat = Q_topk / Z
        P_hat = P_topk_part / T
        logQ_hat = torch.log(Q_hat + eps)
        logP_hat = torch.log(P_hat + eps)

        a = logQ_hat + 1.0 - logP_hat
        mean_a_local = torch.sum(Q_hat * a, dim=-1)
        mean_a = mean_a_local.clone()
        torch.distributed.all_reduce(mean_a, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        grad_topk = (Q_topk / Z) * (a - mean_a.unsqueeze(-1))  # [*, K]

        grad_rkl = torch.zeros_like(Q_full)
        grad_rkl_2d = grad_rkl.view(-1, partition_vocab_size)

        grad_topk_2d = grad_topk.view(-1, topk)
        grad_topk_2d = torch.where(topk_mask.view(-1, topk), grad_topk_2d, torch.zeros_like(grad_topk_2d))

        row2 = torch.arange(grad_rkl_2d.size(0), device=grad_rkl_2d.device).unsqueeze(-1)
        idx_2d = topk_idx_local.view(-1, topk)
        grad_rkl_2d[row2, idx_2d] += grad_topk_2d

        grad_input = kl_ratio * grad_kl + rkl_ratio * grad_rkl
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None, None


def vocab_parallel_kl_rkl_divergence(vocab_parallel_logits, target_topk_logps, target_topk_indices, rkl_ratio: float = 0.1):
    return _VocabParallelWeightedKLRKLDivergence.apply(
        vocab_parallel_logits, target_topk_logps, target_topk_indices, rkl_ratio
    )


# ============================================================
# 4) JSD(beta) with analytic rest term for Q||M outside top-k
# ============================================================
class _VocabParallelJSDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices, beta: float):
        beta = min(max(float(beta), 1e-6), 1.0 - 1e-6)
        one_minus_beta = 1.0 - beta
        eps = 1e-20

        # Student Q_full
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)

        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        vocab_parallel_logits.exp_()
        exp_logits = vocab_parallel_logits
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        Q_full = exp_logits
        Q_full.div_(sum_exp_logits.unsqueeze(-1))  # Q

        # Local vocab range
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        topk_in_vocab = (target_topk_indices >= vocab_start) & (target_topk_indices < vocab_end)

        topk_idx_local = (target_topk_indices - vocab_start).clone()
        topk_idx_local[~topk_in_vocab] = 0

        # Teacher P_topk
        P_topk = torch.exp(target_topk_logps).clone()
        P_topk[~topk_in_vocab] = 0.0

        logP_topk = target_topk_logps.clone()
        logP_topk[~topk_in_vocab] = 0.0

        # Gather Q_topk
        origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)

        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view(origin_shape).clone()
        Q_topk[~topk_in_vocab] = 0.0

        logQ_topk = torch.log(Q_topk + eps)

        # Mix on topk: M_k = beta P_k + (1-beta) Q_k
        M_topk = beta * P_topk + one_minus_beta * Q_topk
        logM_topk = torch.log(M_topk + eps)

        # KL(P||M) topk
        kl_P_M_local = torch.sum(P_topk * (logP_topk - logM_topk), dim=-1)

        # KL(Q||M) topk
        kl_Q_M_topk_local = torch.sum(Q_topk * (logQ_topk - logM_topk), dim=-1)

        # KL(Q||M) rest analytic: for non-topk, M_j=(1-beta)Q_j => Q_j log(1/(1-beta))
        Q_topk_sum_local = Q_topk.sum(dim=-1)
        Q_topk_sum = Q_topk_sum_local.clone()
        torch.distributed.all_reduce(Q_topk_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        log_one_minus_beta = math.log(one_minus_beta)
        Q_rest_sum = 1.0 - Q_topk_sum
        kl_Q_M_rest = Q_rest_sum * (-log_one_minus_beta)

        tmp = beta * kl_P_M_local + one_minus_beta * kl_Q_M_topk_local
        torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        per_token_jsd = tmp + one_minus_beta * kl_Q_M_rest

        ctx.save_for_backward(Q_full, P_topk, topk_idx_local)
        ctx.beta = beta
        return per_token_jsd

    @staticmethod
    def backward(ctx, grad_output):
        Q_full, P_topk, topk_idx_local = ctx.saved_tensors
        beta = ctx.beta
        one_minus_beta = 1.0 - beta
        eps = 1e-20

        partition_vocab_size = Q_full.size(-1)
        topk = topk_idx_local.size(-1)

        # Re-gather Q_topk
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)

        Q_topk_2d = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk_2d.view_as(P_topk)

        topk_mask = P_topk > 0
        Q_topk = torch.where(topk_mask, Q_topk, torch.zeros_like(Q_topk))

        M_topk = beta * P_topk + one_minus_beta * Q_topk
        logQ_topk = torch.log(Q_topk + eps)
        logM_topk = torch.log(M_topk + eps)

        KL_Q_M_topk_local = torch.sum(Q_topk * (logQ_topk - logM_topk), dim=-1)

        Q_topk_sum_local = Q_topk.sum(dim=-1)
        Q_topk_sum = Q_topk_sum_local.clone()
        torch.distributed.all_reduce(Q_topk_sum, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())

        log_one_minus_beta = math.log(one_minus_beta)
        Q_rest_sum = 1.0 - Q_topk_sum
        KL_Q_M_rest = Q_rest_sum * (-log_one_minus_beta)

        KL_Q_M_topk_global = KL_Q_M_topk_local.clone()
        torch.distributed.all_reduce(
            KL_Q_M_topk_global, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        KL_Q_M = KL_Q_M_topk_global + KL_Q_M_rest  # [*]

        # A_j = log(Q_j / M_j); non-topk: -log(1-beta)
        A = torch.full_like(Q_full, -log_one_minus_beta)

        A_topk = logQ_topk - logM_topk
        A_2d = A.view(-1, partition_vocab_size)
        A_topk_2d = A_topk.view(-1, topk)

        idx_2d = topk_idx_local.view(-1, topk)
        row2 = torch.arange(A_2d.size(0), device=A_2d.device).unsqueeze(-1)
        A_2d[row2, idx_2d] = A_topk_2d

        # d/dz JSD = (1-beta) * Q * (A - KL(Q||M))
        grad_input = one_minus_beta * Q_full * (A - KL_Q_M.unsqueeze(-1))
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None, None


def vocab_parallel_jsd_divergence(vocab_parallel_logits, target_topk_logps, target_topk_indices, beta: float = 0.5):
    return _VocabParallelJSDivergence.apply(vocab_parallel_logits, target_topk_logps, target_topk_indices, beta)


# ============================================================
# Unified operator wrapper + factory
# ============================================================
class VocabParallelDistillLoss(torch.nn.Module):
    """
    Unified operator:
      forward(vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices) -> per_token_loss

    Supported names (case-insensitive):
      - "kl"
      - "rkl"
      - "kl_rkl"
      - "jsd"

    Params:
      - rkl_ratio: only used when name == "kl_rkl"
      - beta:      only used when name == "jsd"
    """
    def __init__(self, name: str = "kl", rkl_ratio: float = 0.1, beta: float = 0.5):
        super().__init__()
        self.name = str(name).lower()
        self.rkl_ratio = float(rkl_ratio)
        self.beta = float(beta)

    def forward(self, vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices):
        n = self.name

        if n in ["kl", "forward_kl", "forward-kl"]:
            return vocab_parallel_kl_divergence(vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices)

        if n in ["rkl", "reverse_kl", "reverse-kl"]:
            return vocab_parallel_rkl_divergence(vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices)

        if n in ["kl_rkl", "kl+rkl", "kl_rkl_weighted", "weighted_kl_rkl", "klrkl"]:
            return vocab_parallel_kl_rkl_divergence(
                vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices, rkl_ratio=self.rkl_ratio
            )

        if n in ["jsd", "jensen_shannon", "jensen-shannon", "jensen_shannon_divergence"]:
            return vocab_parallel_jsd_divergence(
                vocab_parallel_logits, teacher_topk_logps, teacher_topk_indices, beta=self.beta
            )

        raise ValueError(f"Unknown distill loss name: {self.name}")


def build_vocab_parallel_distill_loss(loss_cfg: Optional[Any]) -> VocabParallelDistillLoss:
    """
    loss_cfg can be:
      - None
      - dict
      - OmegaConf DictConfig

    Expected fields:
      - name: "kl" | "rkl" | "kl_rkl" | "jsd"
      - rkl_ratio: float (only for kl_rkl)
      - beta: float (only for jsd)
    """
    cfg: Dict[str, Any] = {}
    if loss_cfg is None:
        cfg = {}
    else:
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
            if isinstance(loss_cfg, DictConfig):
                cfg = OmegaConf.to_container(loss_cfg, resolve=True)  # type: ignore
            elif isinstance(loss_cfg, dict):
                cfg = dict(loss_cfg)
            else:
                cfg = {}
        except Exception:
            cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}

    name = str(cfg.get("name", "kl")).lower()
    rkl_ratio = float(cfg.get("rkl_ratio", 0.1))
    beta = float(cfg.get("beta", 0.5))

    return VocabParallelDistillLoss(name=name, rkl_ratio=rkl_ratio, beta=beta)


__all__ = [
    "vocab_parallel_kl_divergence",
    "vocab_parallel_rkl_divergence",
    "vocab_parallel_kl_rkl_divergence",
    "vocab_parallel_jsd_divergence",
    "VocabParallelDistillLoss",
    "build_vocab_parallel_distill_loss",
]
