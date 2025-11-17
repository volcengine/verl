# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Rollout Correction Helper Module

This module provides a complete pipeline to address **off-policy issues** in RL training,
including:
1. Policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)
2. Model update staleness (training on trajectories from older checkpoints)
3. General distribution shifts between data collection and training

Its core capabilities include computing importance sampling (IS) weights,
filtering outlier samples via rejection sampling (RS), and
tracking metrics to diagnose and correct off-policy issues.

## Core Capabilities
1. **Multi-Granularity Aggregation**:
   - Importance Sampling (IS):
        Token-level
        Sequence-level
   - Rejection Sampling (RS):
        Token-level
        Sequence/geometric (sequence-level geometric mean) — supports flexible outlier filtering.
2. **Catastrophic Outlier Veto**:
    Independent per-token veto mechanism — fully reject sequences containing tokens
    with extremely low IS weights (prevents catastrophic updates).
3. **Memory-Efficient Design**:
   - Log-space computations to avoid numerical overflow/underflow.
   - Fixed safety bounds (exp(±20)) for stable exponentiation.
   - Metrics calculated without large intermediate tensors (prevents CUDA OOM).
4. **Comprehensive Metrics Tracking**:
   - IS/RS statistics (mean/max/min, effective sample size ESS, rejection rate).
   - Off-policy diagnostics (KL divergence, perplexity PPL, log PPL difference, χ² divergence).
   - Sequence-level breakdowns (deviation from ideal weights, outlier fraction).


## Key Interfaces & Usage
- compute_rollout_correction_and_rejection_mask(): compute IS weights + rejection mask + veto.
- compute_rollout_correction_weights(): only compute truncated IS weights (for variance
  reduction, no outlier rejection).
- compute_rollout_rejection_mask(): only filter outliers (for sample cleaning, no IS weight
  computation).
- compute_offpolicy_metrics(): called by core functions to calculate off-policy diagnostics
  (KL/PPL/χ²) — no direct external calls needed.

### Integration Notes
- Used in `ray_trainer.py` via `compute_rollout_correction_and_add_to_batch()` (batch training pipeline).
- Used in `dp_actor.py` for distributed worker computations (distributed training scenarios).
- All functions support batch inputs and valid token masking (via `response_mask`).


## References
- "When Speed Kills Stability" (LLM training stability analysis): https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
- Off-policy RL (theoretical basis for IS): https://fengyao.notion.site/off-policy-rl
"""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.workers.config.actor import PolicyLossConfig

# Safety bound to prevent numerical overflow/underflow when exponentiating
# exp(20) ≈ 485 million (upper limit for stable weights), exp(-20) ≈ 2e-9 (lower limit)
SAFETY_BOUND = 20.0


def compute_rollout_rejection_mask(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str = "token",
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rejection mask for outlier handling in off-policy RL training.

    This function identifies and masks outlier tokens/sequences using precomputed log ratios
    (log(π_train / π_rollout)). It supports multiple aggregation levels and uses log-space
    computations for numerical stability.

    Memory-efficient design:
    - Log-space calculations to avoid overflow
    - Fixed safety bounds on exponentiation
    - Metrics computed without large intermediate tensors

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Rejection sampling aggregation level, must be one of:
            - "token": Per-token outlier detection
            - "sequence": Aggregate across entire sequence (product of token ratios)
            - "geometric": Geometric mean across entire sequence
        rollout_rs_threshold: Upper threshold for valid IS weights (required for outlier detection).
        rollout_rs_threshold_lower: Lower threshold for valid IS weights. If None, defaults to 1/upper threshold.

    Returns:
        Tuple containing:
            modified_response_mask: Response mask with outliers masked (0=rejected),
                shape (batch_size, seq_length).
            metrics: Dictionary of rejection sampling metrics (all scalars), including:
                - rollout_rs_mean/max/min: Statistic of IS weights
                - rollout_rs_ratio_fraction_high/low: Fraction of weights exceeding thresholds
                - rollout_rs_masked_fraction: Fraction of tokens rejected (unified for all modes)
                - rollout_rs_seq_masked_fraction: Fraction of sequences rejected (mode-dependent)
    """
    # Validate input parameters
    valid_rs_levels = {"token", "sequence", "geometric"}
    if rollout_rs not in valid_rs_levels:
        raise ValueError(f"Invalid rollout_rs: {rollout_rs}. Must be one of {valid_rs_levels}.")
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be provided for rejection sampling.")

    # Set default lower threshold if not specified (reciprocal of upper threshold)
    upper_threshold = rollout_rs_threshold
    lower_threshold = rollout_rs_threshold_lower if rollout_rs_threshold_lower is not None else 1.0 / upper_threshold

    # Compute IS weights from log ratio (handles different aggregation levels)
    if rollout_rs == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_rs == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to (batch_size, seq_length)

    elif rollout_rs == "geometric":
        # Sequence-level geometric mean: exp(mean(log ratios))
        log_ratio_mean: torch.Tensor = verl_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_mean

        log_ratio_mean_safe: torch.Tensor = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)

    else:
        raise ValueError(f"Unsupported rollout_rs: {rollout_rs}")

    # Generate outlier mask: 1=valid (within [lower, upper] threshold), 0=outlier
    mask: torch.Tensor = (rollout_is_weights >= lower_threshold) & (rollout_is_weights <= upper_threshold)
    mask = mask.float()

    # Compute rejection sampling metrics
    metrics: dict[str, float] = compute_rs_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=upper_threshold,
        rollout_rs_threshold_lower=lower_threshold,
    )

    # Track token-level and sequence-level rejection rates
    # rollout_rs_masked_fraction: fraction of tokens rejected (unified for all modes)
    metrics["rollout_rs_masked_fraction"] = verl_F.masked_mean(1 - mask, response_mask).item()

    # rollout_rs_seq_masked_fraction: fraction of sequences rejected (mode-dependent)
    if rollout_rs == "token":
        # Token-level aggregation: sequence is rejected if any token is rejected
        seq_has_masked: torch.Tensor = verl_F.masked_sum(1 - mask, response_mask, axis=-1) > 0
        metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
    else:
        # Sequence-level aggregation: check first token's mask (all tokens in sequence have same mask)
        metrics["rollout_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()

    # Apply rejection mask to original response mask
    modified_response_mask: torch.Tensor = response_mask * mask

    return modified_response_mask, metrics


def compute_rs_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str,
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
) -> dict[str, float]:
    """Compute comprehensive metrics for rejection sampling.

    This function calculates statistics for IS weights used in rejection sampling,
    balancing numerical stability (using clamped weights) and accuracy (using log-space
    for threshold checks).

    Args:
        rollout_is_weights: Clamped IS weights (π_train / π_rollout),
            shape (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio of training to rollout probabilities (unclamped),
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Rejection sampling aggregation level (matches compute_rollout_rejection_mask).
        rollout_rs_threshold: Upper threshold for valid IS weights.
        rollout_rs_threshold_lower: Lower threshold for valid IS weights.

    Returns:
        Dictionary of rejection sampling metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rollout_is_weights.device

    # Precompute log thresholds for accurate threshold checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    if rollout_rs in ["sequence", "geometric"]:
        # Sequence-level aggregation: use log-space for accurate max/min/threshold checks
        # True max/min (unclamped) converted with safety bounds
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_rs_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_rs_min"] = torch.exp(log_min).item()

        # Mean uses clamped weights to avoid overflow
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        # Both sequence and geometric modes operate at sequence level (batch_size, 1)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_rs_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from clamped weights
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_rs_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_rs_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_rs_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens first)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_rs_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_rs_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        # Clamp weights to threshold range to avoid squaring extreme values
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_rs_threshold_lower, max=rollout_rs_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        # Variance = E[X²] - (E[X])² (masked to valid tokens)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_rs_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_rs_std"] = 0.0

    # Compute Effective Sample Size (ESS) for IS weights
    # ESS = 1 / E[(w_i / E[w_i])²] (using clamped weights for stability)
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_rs_threshold_lower, max=rollout_rs_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_rs_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        # Mean weight per sequence (masked to valid tokens)
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_rs_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_rs_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_rs_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_rs_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_rs_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_rs_seq_fraction_high"] = (seq_mean_weights > rollout_rs_threshold).float().mean().item()
        metrics["rollout_rs_seq_fraction_low"] = (seq_mean_weights < rollout_rs_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute importance sampling weights to correct for off-policy distribution shifts.

    This function calculates IS weights (π_train / π_rollout) using log ratios for numerical stability.
    It supports multiple aggregation levels and truncates extreme weights to prevent training instability.

    Key design:
    - Log-space computations to avoid overflow
    - Truncation of extreme weights (TIS: Truncated Importance Sampling)
    - Optional batch normalization (normalize to mean=1.0)
    - Metrics tracking for weight distribution analysis

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level, must be one of:
            - "token": Per-token weights (biased, low variance)
            - "sequence": Per-sequence weight (product of tokens; unbiased, high variance)
        rollout_is_threshold: Upper threshold for truncating extreme weights (e.g., 2.0),
            default 2.0.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch,
            default False.

    Returns:
        Tuple containing:
            rollout_is_weights: Truncated IS weights (masked to zero for padding tokens),
                shape (batch_size, seq_length). If batch_normalize=True, normalized to mean=1.0.
            metrics: Dictionary of IS weight metrics (all scalars), including:
                - rollout_is_mean/max/min: Statistic of weights (before batch normalization)
                - rollout_is_eff_sample_size: Effective sample size (ESS)
                - rollout_is_seq_*: Sequence-level weight statistics
                - rollout_is_batch_norm_factor: Normalization factor (only if batch_normalize=True)
    """
    # Validate input parameters
    valid_is_levels = {"token", "sequence"}
    if rollout_is not in valid_is_levels:
        raise ValueError(f"Invalid rollout_is: {rollout_is}. Must be one of {valid_is_levels}.")
    if rollout_is_threshold <= 0:
        raise ValueError(f"rollout_is_threshold must be positive, got {rollout_is_threshold}.")

    # Compute IS weights from log ratio (handles different aggregation levels)
    if rollout_is == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_is == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to sequence length

    else:
        raise ValueError(f"Unsupported rollout_is: {rollout_is}")

    # Zero out weights for padding tokens using response mask
    rollout_is_weights = rollout_is_weights * response_mask

    # Compute IS weight metrics (BEFORE truncation to get accurate fraction_high/low)
    metrics: dict[str, float] = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
    )

    # Truncate extreme weights (TIS: Truncated Importance Sampling)
    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)

    # Detach weights to prevent gradient flow (mathematically required by IS theory)
    # IS weights change the measure, not the objective. See §3.2.2 in docs/algo/rollout_corr_math.md
    rollout_is_weights = rollout_is_weights.detach()

    # Apply batch normalization if requested
    if rollout_is_batch_normalize:
        # Compute mean based on aggregation level
        if rollout_is == "token":
            # Token-level: normalize over all token weights
            weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask)
        elif rollout_is == "sequence":
            # Sequence-level: normalize over sequence weights (one weight per sequence)
            # For each sequence, compute mean over valid tokens (they all have the same weight)
            # then average across sequences
            seq_weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)  # (batch_size,)
            weights_mean = seq_weights_mean.mean()
        else:
            raise ValueError(f"Unsupported rollout_is: {rollout_is}")

        # Normalize to mean=1.0 (avoid division by zero)
        if weights_mean > 1e-8:
            rollout_is_weights = rollout_is_weights / weights_mean
            metrics["rollout_is_batch_norm_factor"] = weights_mean.item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return rollout_is_weights, metrics


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
    """Compute comprehensive metrics for truncated importance sampling weights.

    This function calculates statistics for truncated IS weights (TIS), using log-space
    for accurate threshold checks and clamped weights for stable mean/std calculations.

    Args:
        rollout_is_weights: Truncated IS weights (π_train / π_rollout),
            shape (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio of training to rollout probabilities (unclamped),
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level (matches compute_rollout_correction_weights).
        rollout_is_threshold: Upper threshold for truncated IS weights.

    Returns:
        Dictionary of IS weight metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rollout_is_weights.device
    # Default lower threshold (reciprocal of upper threshold)
    rollout_is_threshold_lower: float = 1.0 / rollout_is_threshold

    # Precompute log thresholds for accurate checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    if rollout_is == "sequence":
        # Sequence-level aggregation: use log-space for unclamped stats
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_is_min"] = torch.exp(log_min).item()

        # Mean uses truncated weights to avoid overflow
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_is_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from truncated weights
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_is_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_is_threshold_lower, max=rollout_is_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_is_std"] = 0.0

    # Compute Effective Sample Size (ESS) for truncated weights
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_is_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_is_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean().item()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: Optional[float] = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = 2.0,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Optional[DataProto], torch.Tensor, dict[str, float]]:
    """Unified interface for computing IS weights and rejection masks.

    This function combines IS weight calculation (truncated) and rejection sampling (masked)
    into a single pipeline. It also applies a per-token veto for catastrophic outliers
    (sequences with extremely low token ratios are fully rejected).

    Key design:
    - Separation of IS weights (for variance reduction) and rejection masks (for sample filtering)
    - Veto mechanism for catastrophic sequences (applied independently of other modes)
    - Comprehensive metrics tracking for mismatch diagnosis

    Args:
        old_log_prob: Log probabilities from the training policy (e.g., FSDP FP32),
            shape (batch_size, seq_length).
        rollout_log_prob: Log probabilities from the rollout policy (e.g., vLLM BF16),
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_is: IS weight aggregation level (see compute_rollout_correction_weights for options).
            Set to None to disable IS weight computation.
        rollout_is_threshold: Upper threshold for truncated IS weights (used if rollout_is is set),
            default 2.0.
        rollout_rs: Rejection sampling aggregation level (see compute_rollout_rejection_mask for options).
            Set to None to disable rejection sampling.
        rollout_rs_threshold: Upper threshold for rejection sampling. Required if rollout_rs is enabled.
            Default 2.0.
        rollout_rs_threshold_lower: Lower threshold for rejection sampling (used if rollout_rs is set).
            Defaults to 1/rollout_rs_threshold if None.
        rollout_token_veto_threshold: Minimum allowed token-level IS weight. Sequences containing
            any token below this threshold are fully rejected. Set to None to disable veto.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch.
            Default: False.

    Returns:
        Tuple containing:
            rollout_is_weights_proto: DataProto with IS weights (None if rollout_is is None),
                key "rollout_is_weights", shape (batch_size, seq_length).
            modified_response_mask: Response mask with rejection sampling and veto applied,
                shape (batch_size, seq_length).
            metrics: Dictionary of all metrics (prefixed with "rollout_corr/"), including:
                - IS weight statistics
                - Rejection sampling rates
                - Veto statistics
                - Policy mismatch metrics (KL, PPL, etc.)
    """
    # Validate input masks
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")
    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            f"old_log_prob shape {old_log_prob.shape} does not match rollout_log_prob shape {rollout_log_prob.shape}."
        )
    if old_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"log_prob shape {old_log_prob.shape} does not match response_mask shape {response_mask.shape}."
        )

    # Step 1: Compute log ratio (log(π_train / π_rollout))
    log_ratio: torch.Tensor = old_log_prob - rollout_log_prob
    device: torch.device = log_ratio.device
    metrics: dict[str, float] = {}

    # Step 2: Compute IS weights (if enabled)
    rollout_is_weights: Optional[torch.Tensor] = None
    if rollout_is is not None and rollout_is_threshold is not None:
        rollout_is_weights, is_metrics = compute_rollout_correction_weights(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
        )
        metrics.update(is_metrics)

    # Step 3: Compute rejection mask (if enabled)
    modified_response_mask: torch.Tensor = response_mask.clone()
    if rollout_rs is not None:
        if rollout_rs_threshold is None:
            raise ValueError(
                "rollout_rs_threshold must be explicitly provided when rollout_rs is enabled. "
                "Set rollout_rs_threshold to the desired threshold value."
            )
        modified_response_mask, rs_metrics = compute_rollout_rejection_mask(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
            rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        )
        metrics.update(rs_metrics)

    # Step 4: Apply per-token veto (reject sequences with catastrophic tokens)
    if rollout_token_veto_threshold is not None:
        if rollout_token_veto_threshold <= 0:
            raise ValueError(f"rollout_token_veto_threshold must be positive, got {rollout_token_veto_threshold}.")

        # Compute log threshold for numerical stability
        log_veto_threshold: torch.Tensor = torch.log(torch.tensor(rollout_token_veto_threshold, device=device))
        # Identify catastrophic tokens (log ratio below threshold + valid mask)
        catastrophic_tokens: torch.Tensor = (log_ratio < log_veto_threshold) & response_mask.bool()
        # Check if sequence contains any catastrophic token
        has_catastrophic: torch.Tensor = catastrophic_tokens.any(dim=-1, keepdim=True)
        # Create veto mask (0=reject sequence, 1=keep)
        veto_mask: torch.Tensor = (~has_catastrophic).float()

        # Track veto metrics
        metrics["rollout_is_veto_fraction"] = has_catastrophic.float().mean().item()
        metrics["rollout_is_catastrophic_token_fraction"] = verl_F.masked_mean(
            catastrophic_tokens.float(), response_mask
        ).item()

        # Apply veto to response mask (overrides previous rejection)
        modified_response_mask = modified_response_mask * veto_mask
    else:
        # Add placeholder metrics if veto is disabled
        metrics["rollout_is_veto_fraction"] = 0.0
        metrics["rollout_is_catastrophic_token_fraction"] = 0.0

    # Step 5: Compute off-policy metrics (KL, PPL, χ², etc.)
    offpolicy_metrics: dict[str, float] = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    metrics.update(offpolicy_metrics)

    # Step 6: Add "rollout_corr/" prefix to all metrics for logging consistency
    metrics_scalar: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_scalar[f"rollout_corr/{key}"] = value

    # Step 7: Wrap IS weights in DataProto for consistency with API
    rollout_is_weights_proto: Optional[DataProto] = None
    if rollout_is_weights is not None:
        rollout_is_weights_proto = DataProto.from_dict(tensors={"rollout_is_weights": rollout_is_weights})

    return rollout_is_weights_proto, modified_response_mask, metrics_scalar


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute off-policy diagnostic metrics (helper function).

    This helper function operates on raw tensors and is used internally by:
    - compute_rollout_correction_and_rejection_mask() in this module (automatically included)
    - Tests (test_rollout_corr.py, test_rollout_corr_integration.py)

    These metrics help diagnose the off-policy gap between rollout and training policies,
    which can arise from:
    - Policy mismatch (e.g., vLLM BF16 vs FSDP FP32)
    - Model staleness (training on trajectories from older checkpoints)
    - General distribution shifts

    Key metrics:
    - kl: Direct KL divergence estimator KL(π_rollout || π_training)
    - k3_kl: K3 KL estimator for stability (more stable for small KL)
    - training_ppl: Perplexity of training policy
    - rollout_ppl: Perplexity of rollout policy
    - log_ppl_diff: Difference in log perplexities
    - ppl_ratio: Ratio of training PPL to rollout PPL
    - chi2_token: Token-level χ² divergence E[ρ²] - 1
    - chi2_seq: Sequence-level χ² divergence E[(∏ρ_t)²] - 1

    Args:
        old_log_prob: Log probabilities from training policy, shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy, shape (batch_size, seq_length)
        response_mask: Mask for valid tokens, shape (batch_size, seq_length)

    Returns:
        Dictionary of off-policy metrics (without prefix)
    """
    # Validate that we have at least one valid token
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics = {}

    # 1. Training policy perplexity (always available)
    # Formula: exp(-1/|T| * Σ log π_training(y_t|y_<t))
    # where |T| is the number of tokens generated by the model
    mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
    metrics["training_ppl"] = training_ppl.detach().item()

    # Also log log-ppl for easier analysis (avoids exponential scale)
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    # 2. Compute rollout off-policy metrics (only if rollout_log_probs available)
    if rollout_log_prob is not None:
        # 2a. kl: Direct estimator for KL(π_rollout || π_training)
        # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
        # Positive value means rollout policy is more confident than training policy
        metrics["kl"] = verl_F.masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()

        # 2b. k3_kl: K3 estimator for KL(π_rollout || π_training)
        # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
        # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
        log_ratio = old_log_prob - rollout_log_prob
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = verl_F.masked_mean(k3_kl_matrix, response_mask).detach().item()

        # 2c. Rollout policy perplexity
        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL
        metrics["rollout_ppl"] = rollout_ppl.detach().item()
        metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

        # 2d. Log PPL difference (sequence-level perplexity difference)
        # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        # Since ppl = exp(-log_prob), we have:
        #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
        # Positive value means training assigns lower probability (higher PPL) than rollout
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()

        # 2e. PPL ratio (how much higher is training PPL vs rollout PPL)
        # IMPORTANT: Compute per-sequence ratio first, then average
        # For numerical stability, compute in log space using log_ppl_diff
        # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
        # This is the inverse of geometric IS: ppl_ratio_i = 1 / geometric_is_i for each sequence
        ppl_ratio = torch.exp(log_ppl_diff).mean()  # mean(exp(log_ppl_diff)) = mean(ppl_ratio_i)
        metrics["ppl_ratio"] = ppl_ratio.detach().item()

        # 2f. Chi-squared divergence: χ²(π_training || π_rollout) = E_μ[ρ²] - 1
        # where ρ = π_training / π_rollout and μ = π_rollout (rollout distribution)
        # This measures the variance of importance sampling weights
        # Token-level: E_token[ρ²] - 1 (averaged over all tokens)
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)  # ρ = π_training / π_rollout (token-level)
        rho_squared_token = rho_token.square()
        chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0
        metrics["chi2_token"] = chi2_token.detach().item()

        # Sequence-level: E_seq[(Π ρ_t)²] - 1 = E_seq[exp(2 * Σ log ρ_t)] - 1
        log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)  # Σ log ρ_t per sequence
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)  # (Π ρ_t)²
        chi2_seq = rho_squared_seq.mean() - 1.0
        metrics["chi2_seq"] = chi2_seq.detach().item()

    return metrics


def compute_rollout_correction_and_add_to_batch(
    batch: DataProto, rollout_corr_config: RolloutCorrectionConfig
) -> tuple[DataProto, dict]:
    """Compute rollout correction weights and apply rejection sampling.

    Computes importance sampling weights to correct for off-policy issues between
    rollout and training policies. Applies rejection sampling by modifying response_mask.
    Always updates response_mask; conditionally adds IS weights.

    Key behavior:
    - response_mask: ALWAYS updated with rejection (veto + optional RS excluded from training)
    - rollout_is_weights: Added to batch ONLY if rollout_is parameter is set

    This separation ensures:
    - Rejection works independently of IS weight application
    - Metrics can be monitored before enabling IS weight correction

    Args:
        batch: DataProto with old_log_probs, rollout_log_probs, response_mask

    Returns:
        Tuple of (updated_batch, metrics):
            updated_batch: Batch with modified response_mask (always) and rollout_is_weights (if enabled)
            metrics: Dict of IS and off-policy metrics, all with "rollout_corr/" prefix

    Note:
        The implementation is copied from szrlee <szrlee@gmail.com>.
    """
    # Get new API parameters directly from config
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # Compute IS weights and get modified response_mask
    rollout_is_weights, modified_response_mask, rollout_corr_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=batch.batch["response_mask"],
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
    )

    # ALWAYS update response_mask with rejection applied
    batch.batch["response_mask"] = modified_response_mask

    # Add IS weights to batch if computed
    if rollout_is_weights is not None:
        batch = batch.union(rollout_is_weights)

    return batch, rollout_corr_metrics


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout correction metrics from log probabilities during training.

    This function is used in the actor to compute metrics using the CURRENT policy
    log probabilities versus rollout log probabilities, allowing tracking of the
    off-policy gap as training progresses.

    It computes off-policy diagnostic metrics (KL, PPL, χ²) from log probabilities.

    Args:
        log_prob: Current policy log probabilities, shape (batch_size, seq_length)
        rollout_log_prob: Rollout policy log probabilities, shape (batch_size, seq_length)
        response_mask: Valid token mask, shape (batch_size, seq_length)

    Returns:
        Dictionary of metrics with "rollout_corr/" prefix
    """
    # Compute off-policy diagnostic metrics
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    # Add rollout_corr/ prefix to all metrics
    metrics_with_prefix = {}
    for key, value in offpolicy_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_with_prefix[f"rollout_corr/{key}"] = value

    return metrics_with_prefix


def apply_rollout_correction(
    batch: DataProto,
    rollout_corr_config: Optional[RolloutCorrectionConfig] = None,
    policy_loss_config: PolicyLossConfig = None,
) -> None:
    """
    BYPASS MODE: Use rollout_log_probs as old_log_probs
    Skips expensive actor forward pass for old_log_prob computation

    Two sub-modes (controlled by use_policy_gradient):
    1. Bypass + PPO loss (use_policy_gradient=False, default):
       - Uses standard PPO loss function with old_log_prob=rollout_log_prob
       - PPO clips ratio π_θ/π_rollout instead of π_θ/π_old

    2. Bypass + Policy Gradient loss (use_policy_gradient=True):
       - Uses compute_policy_loss_with_rollout_correction()
       - Policy gradient (REINFORCE-style) with IS/RS correction applied
       - No PPO clipping

    Note:
        The implementation is copied from szrlee <szrlee@gmail.com>.
    """
    from omegaconf import open_dict

    if "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "bypass_mode=True requires rollout_log_probs in batch. "
            "Ensure rollout worker is configured to calculate_log_probs=true."
        )

    # Use rollout log probs as old log probs (zero-cost substitution)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    with open_dict(policy_loss_config):
        # Always pass rollout_correction config to actor for metrics computation
        policy_loss_config["rollout_correction"] = rollout_corr_config

    # Check if policy gradient loss mode is enabled
    use_policy_gradient = rollout_corr_config.get("use_policy_gradient", False)

    if use_policy_gradient:
        # Policy gradient mode: Configure actor to use rollout_correction loss function
        # This will use compute_policy_loss_with_rollout_correction (no PPO clipping)
        policy_loss_config["loss_mode"] = "rollout_correction"
