# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Shared head/sample scoring for FSDP's full local vocabulary.

This is a separate backend path: the general auto-correction helper continues to
consume full-vocabulary-normalized head probabilities.
"""

from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _vocabulary_gradient(
        logits,
        gradient,
        maxima,
        denominators,
        coefficients,
        aligned,
        vocab: tl.constexpr,
        logits_row_stride: tl.constexpr,
        logits_col_stride: tl.constexpr,
        grad_row_stride: tl.constexpr,
        grad_col_stride: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        in_vocab = columns < vocab
        if tl.load(aligned + row):
            values = tl.full((block,), 0.0, tl.float32)
        else:
            values = tl.load(
                logits + row * logits_row_stride + columns * logits_col_stride,
                mask=in_vocab,
                other=-float("inf"),
            ).to(tl.float32)
            maximum = tl.load(maxima + row)
            denominator = tl.load(denominators + row)
            coefficient = tl.load(coefficients + row)
            values = -coefficient * tl.exp(values - maximum) / denominator
        tl.store(gradient + row * grad_row_stride + columns * grad_col_stride, values, mask=in_vocab)


class _SharedScoreCenteringScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, head_ids, labels, sampler_head, aligned, weight_fn, chunk_size):
        n = logits.shape[0]
        chunk_size = chunk_size if chunk_size > 0 else max(n, 1)
        correction = torch.empty(n, device=logits.device, dtype=torch.float32)
        sampled = torch.empty_like(correction)
        train_mass = torch.empty_like(correction)
        maxima = torch.empty_like(correction)
        denominators = torch.empty_like(correction)
        residuals = torch.empty(head_ids.shape, device=logits.device, dtype=torch.float32)
        head_probabilities = torch.empty_like(residuals)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            raw_head = work.gather(-1, head_ids[start:end])
            raw_sample = work.gather(-1, labels[start:end].unsqueeze(-1)).squeeze(-1)
            maximum = work.max(-1).values
            work.sub_(maximum.unsqueeze(-1)).exp_()
            denominator = work.sum(-1)
            del work
            log_denominator = denominator.log()
            full_head_lp = (raw_head - maximum.unsqueeze(-1)) - log_denominator.unsqueeze(-1)

            complete = aligned[start:end]
            q_lp = sampler_head[start:end].float()
            finite = q_lp.isfinite()
            # A fallback row need not contain a valid complete support. Keep its
            # unselected support arithmetic finite, as in auto_correction.
            support = torch.where(complete.unsqueeze(-1), finite, torch.ones_like(finite))
            support_logits = raw_head.masked_fill(~support, float("-inf"))
            support_maximum = support_logits.max(-1).values
            support_log_denominator = (support_logits - support_maximum.unsqueeze(-1)).exp().sum(-1).log()
            support_lp = (
                (raw_head - support_maximum.unsqueeze(-1)) - support_log_denominator.unsqueeze(-1)
            ).masked_fill(~support, 0.0)
            head_lp = torch.where(complete.unsqueeze(-1), support_lp, full_head_lp)
            p = head_lp.exp() * (~complete.unsqueeze(-1) | finite)
            q = q_lp.exp()
            ratio = torch.where(finite, (head_lp - q_lp).exp(), torch.ones_like(q))
            weighted = q * weight_fn(ratio)
            p_mass, q_mass = p.sum(-1), q.sum(-1)
            rho = (1 - q_mass).clamp_min(1e-6) / (1 - p_mass).clamp_min(1e-6)
            alpha = rho * weight_fn(1 / rho)
            residual = torch.where(
                complete.unsqueeze(-1),
                weighted - weighted.sum(-1, keepdim=True) * p,
                weighted - alpha.unsqueeze(-1) * p,
            )
            correction[start:end] = (residual * head_lp).sum(-1)
            sampled[start:end] = torch.where(
                complete,
                (raw_sample - support_maximum) - support_log_denominator,
                (raw_sample - maximum) - log_denominator,
            )
            full_support_log_mass = (support_maximum - maximum) + support_log_denominator - log_denominator
            train_mass[start:end] = torch.where(complete, full_support_log_mass.exp(), full_head_lp.exp().sum(-1))
            maxima[start:end] = maximum
            denominators[start:end] = denominator
            residuals[start:end] = residual
            head_probabilities[start:end] = p
        ctx.save_for_backward(logits, head_ids, labels, aligned, residuals, head_probabilities, maxima, denominators)
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(train_mass)
        return correction, sampled, train_mass

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_correction, grad_sampled, grad_train_mass):
        logits, ids, labels, aligned, residual, head_p, maxima, denominators = ctx.saved_tensors
        n, vocab = logits.shape
        if grad_correction is None:
            grad_correction = torch.zeros(n, device=logits.device, dtype=torch.float32)
        if grad_sampled is None:
            grad_sampled = torch.zeros(n, device=logits.device, dtype=torch.float32)
        grad_head = residual * grad_correction.float().unsqueeze(-1)
        grad_sampled = grad_sampled.float()
        coefficients = grad_head.sum(-1) + grad_sampled
        grad_head = grad_head - torch.where(
            aligned.unsqueeze(-1), coefficients.unsqueeze(-1) * head_p, torch.zeros_like(grad_head)
        )
        grad_input = torch.empty_like(logits)
        if logits.is_cuda and triton is not None and n:
            block = 8192
            _vocabulary_gradient[(n, triton.cdiv(vocab, block))](
                logits,
                grad_input,
                maxima,
                denominators,
                coefficients,
                aligned,
                vocab,
                logits.stride(0),
                logits.stride(1),
                grad_input.stride(0),
                grad_input.stride(1),
                block,
                num_warps=4,
            )
        else:
            grad_input.zero_()
            for start in range(0, n, ctx.chunk_size):
                end = min(start + ctx.chunk_size, n)
                rows = (~aligned[start:end]).nonzero().squeeze(-1) + start
                work = logits.index_select(0, rows).to(dtype=torch.float32, copy=True)
                work.sub_(maxima[rows].unsqueeze(-1)).exp_().div_(denominators[rows].unsqueeze(-1))
                work.mul_(-coefficients[rows].unsqueeze(-1))
                grad_input.index_copy_(0, rows, work.to(logits.dtype))
        # Reduce duplicate head/sample ids within the small K+1 matrix. Then
        # overwrite their dense gradient entries after one fp32 accumulation,
        # without allocating a full-vocabulary fp32 gradient workspace.
        sparse_ids = torch.cat((ids, labels.unsqueeze(-1)), dim=-1)
        sparse_values = torch.cat((grad_head, grad_sampled.unsqueeze(-1)), dim=-1)
        sorted_ids, order = sparse_ids.sort(-1)
        sorted_values = sparse_values.gather(-1, order)
        starts = torch.ones_like(sorted_ids, dtype=torch.bool)
        starts[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
        groups = starts.long().cumsum(-1) - 1
        compact_ids = torch.zeros_like(sorted_ids).scatter_(-1, groups, sorted_ids)
        compact_values = torch.zeros_like(sorted_values).scatter_add_(-1, groups, sorted_values)
        valid = torch.arange(sparse_ids.shape[-1], device=logits.device).unsqueeze(0) < starts.sum(-1, keepdim=True)
        # Unused slots repeat the first group's exact id/value, so repeated
        # stores write an identical result rather than overwriting a live id.
        compact_ids = torch.where(valid, compact_ids, compact_ids[:, :1])
        compact_values = torch.where(valid, compact_values, compact_values[:, :1])
        base = logits.gather(-1, compact_ids).float()
        base.sub_(maxima.unsqueeze(-1)).exp_().div_(denominators.unsqueeze(-1))
        base.mul_(-coefficients.unsqueeze(-1))
        base = torch.where(aligned.unsqueeze(-1), torch.zeros_like(base), base)
        grad_input.scatter_(-1, compact_ids, (base + compact_values).to(logits.dtype))
        return grad_input, None, None, None, None, None, None


def score_centering_shared_scores_from_logits(
    logits: torch.Tensor,
    head_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    support_complete: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor],
    chunk_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Compute centering and sampled scores in one full-local-vocabulary FSDP path.

    Verified complete rows normalize within the finite sampler head. Other rows
    retain the full-vocabulary trainer-tail proxy. The full-vocabulary head-mass
    diagnostic remains detached; it still requires a chunked O(NV) forward scan.
    This helper is not valid for an unmaterialized TP vocabulary shard.
    """
    if logits.ndim != 2 or head_ids.ndim != 2 or logits.shape[0] != head_ids.shape[0]:
        raise ValueError("Expected logits (N, V) and head ids (N, K).")
    n = logits.shape[0]
    if sampler_head_log_probs.shape != head_ids.shape or not head_ids.shape[-1]:
        raise ValueError("Sampler probabilities must match a nonempty head.")
    if sampled_token_ids.shape != (n,) or support_complete.shape != (n,):
        raise ValueError("Sampled ids and complete-support receipts must have shape (N,).")
    aligned = support_complete.bool().contiguous()
    correction, sampled, train_mass = _SharedScoreCenteringScores.apply(
        logits,
        head_ids.long(),
        sampled_token_ids.long(),
        sampler_head_log_probs,
        aligned,
        weight_fn,
        chunk_size,
    )
    with torch.no_grad():
        sampler_mass = sampler_head_log_probs.float().exp().sum(-1)
        tail_mass = (1 - sampler_mass).clamp(min=0)
    return {
        "sc_correction": correction,
        "log_probs": sampled,
        "sc_sampler_head_mass": sampler_mass,
        "sc_train_head_mass": train_mass,
        # Sampled scores are already normalized in the selected per-row domain.
        "sc_log_support_mass": torch.zeros_like(sampled),
        "sc_support_aligned": aligned.float(),
        "sc_sampler_tail_mass": tail_mass,
        "sc_support_unknown": ((~aligned) & (tail_mass <= 1e-6)).float(),
    }
