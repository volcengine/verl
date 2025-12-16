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
- "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch": https://richardli.xyz/rl-collapse
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
    group_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute hard trust region mask using IS ratios or KL divergence estimators.

    This function enforces a hard trust region constraint by masking tokens/sequences
    where the estimated divergence (between training and rollout policies) exceeds
    a threshold. Unlike PPO's soft clipping, this provides a hard boundary.

    IS Ratio Modes (ideal = 1.0, threshold is [lower, upper] ratio bounds):
    - token: Per-token IS ratio
    - sequence: Product of token ratios (sequence-level IS)
    - geometric: Geometric mean of token ratios exp(E[log(r)]) (length-normalized)

    KL Divergence Estimators (ideal = 0.0, threshold is max allowed divergence):
    - K1 (k1/group_k1): |E[log(r)]| where r = π_train/π_rollout
      Absolute mean log ratio, always >= 0
    - K3 (k3/group_k3): E[r - log(r) - 1], always >= 0
      More stable than K1 for small KL values

    Memory-efficient design:
    - Log-space calculations to avoid overflow
    - Fixed safety bounds on exponentiation
    - Metrics computed without large intermediate tensors

    Args:
        log_ratio: Log ratio of training policy probability to rollout policy probability,
            shape (batch_size, seq_length).
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Trust region estimation level, must be one of:
            - "token": Per-token IS ratio (ideal = 1.0)
            - "sequence": Sequence-level IS ratio (ideal = 1.0)
            - "geometric": Geometric mean IS ratio exp(E[log(r)]) (ideal = 1.0)
            - "k1": Sequence-level K1 divergence |E[log(r)]| (ideal = 0.0)
            - "k3": Sequence-level K3 divergence E[r - log(r) - 1] (ideal = 0.0)
            - "group_k1": Group-level K1 divergence (ideal = 0.0)
            - "group_k3": Group-level K3 divergence (ideal = 0.0)
        rollout_rs_threshold: Trust region upper threshold (required).
            For ratio modes (token/sequence/geometric): max allowed ratio (e.g., 2.0)
            For divergence modes (k1/k3/group_k1/group_k3): max allowed divergence (e.g., 0.1)
        rollout_rs_threshold_lower: Trust region lower threshold.
            For ratio modes: min allowed ratio. Defaults to 1/upper_threshold.
            For divergence modes: ignored (divergence >= 0 always).
        group_indices: Optional tensor of group indices, shape (batch_size,).
            Required for "group_k1" or "group_k3" modes.

    Returns:
        Tuple containing:
            modified_response_mask: Response mask with trust region violations masked (0=rejected),
                shape (batch_size, seq_length).
            metrics: Dictionary of trust region metrics (all scalars), including:
                - rollout_rs_k1/k3_mean/max/min: KL divergence statistics (for divergence modes)
                - rollout_rs_fraction_high/low: Fraction exceeding thresholds
                - rollout_rs_masked_fraction: Fraction of tokens masked
                - rollout_rs_seq_masked_fraction: Fraction of sequences masked
    """
    # Validate input parameters
    valid_rs_levels = {"token", "sequence", "geometric", "k1", "k3", "group_k1", "group_k3"}
    if rollout_rs not in valid_rs_levels:
        raise ValueError(f"Invalid rollout_rs: {rollout_rs}. Must be one of {valid_rs_levels}.")
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be provided for rejection sampling.")

    # Handle empty batch gracefully (avoid errors from group_indices.max() etc.)
    if log_ratio.shape[0] == 0:
        return response_mask, {}

    # Set default lower threshold if not specified (reciprocal of upper threshold)
    upper_threshold = rollout_rs_threshold
    lower_threshold = rollout_rs_threshold_lower if rollout_rs_threshold_lower is not None else 1.0 / upper_threshold

    # Compute RS statistic from log ratio (handles different aggregation levels)
    # Note: rs_statistic is the value used for rejection thresholding
    # - For ratio modes (token/sequence): IS ratio exp(log_ratio), ideal = 1.0
    # - For divergence modes (geometric/k3/group_k1/group_k3): divergence value >= 0, ideal = 0.0
    if rollout_rs == "token":
        # Per-token ratio: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rs_statistic: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_rs == "sequence":
        # Sequence-level ratio: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rs_statistic = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to (batch_size, seq_length)

    elif rollout_rs == "geometric":
        # Geometric mean of token ratios at sequence level: exp(E[log(r)])
        # This is equivalent to the geometric mean of per-token IS ratios
        # Ideal = 1.0, threshold is [lower, upper] ratio bounds
        log_ratio_mean: torch.Tensor = verl_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_mean  # Store log-space for metrics

        log_ratio_mean_safe: torch.Tensor = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rs_statistic = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)  # Geometric mean ratio

    elif rollout_rs == "k1":
        # K1 divergence at sequence level: |E[log(r)]|
        # Use absolute value for symmetric divergence (always >= 0, ideal = 0)
        # This equals KL(π_rollout || π_train) in expectation (the reverse KL)
        log_ratio_mean: torch.Tensor = verl_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        k1_div: torch.Tensor = log_ratio_mean.abs()  # |E[log(r)]|
        log_ratio_for_metrics = k1_div  # Store K1 divergence for metrics

        # For K1 divergence, use |mean log ratio| for thresholding
        # K1 >= 0 (due to abs), threshold is max allowed K1 divergence
        rs_statistic = k1_div.expand_as(log_ratio)

    elif rollout_rs == "k3":
        # K3 divergence at sequence level: E[r - log(r) - 1]
        # where r = π_train/π_rollout = exp(log_ratio)
        # K3 >= 0 per token (equals 0 when r=1), equals KL(π_rollout || π_train) in expectation
        # More stable than K1 because each token contribution is non-negative
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        r = torch.exp(log_ratio_safe)  # Token-level ratio: r = exp(log_ratio_safe)
        k3_token = r - log_ratio_safe - 1  # K3 = r - log(r) - 1, using log(r) = log_ratio_safe

        # Sequence-level K3: mean of token K3 values
        k3_seq: torch.Tensor = verl_F.masked_mean(k3_token, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = k3_seq  # Store K3 values for metrics

        # For K3, use K3 divergence value for thresholding (NOT an IS ratio)
        # K3 >= 0, threshold is max allowed K3 divergence
        rs_statistic = k3_seq.expand_as(log_ratio)

    elif rollout_rs == "group_k1":
        # Group-level K1 divergence: |group_mean(E[log(r)])|
        # Reject entire groups of sequences together based on K1 divergence
        if group_indices is None:
            raise ValueError("group_indices must be provided when rollout_rs='group_k1'.")
        if torch.any(group_indices < 0):
            raise ValueError("`group_indices` must contain non-negative values.")

        # First compute sequence-level mean log ratio
        log_ratio_mean: torch.Tensor = verl_F.masked_mean(log_ratio, response_mask, axis=-1)  # (batch_size,)

        # Vectorized group aggregation using scatter operations
        # Get the number of groups (max index + 1)
        num_groups = group_indices.max().item() + 1
        device = log_ratio_mean.device

        # Compute group sums and counts using scatter_add
        group_sums = torch.zeros(num_groups, device=device, dtype=log_ratio_mean.dtype)
        group_counts = torch.zeros(num_groups, device=device, dtype=log_ratio_mean.dtype)
        group_sums.scatter_add_(0, group_indices, log_ratio_mean)
        group_counts.scatter_add_(0, group_indices, torch.ones_like(log_ratio_mean))

        # Compute group means (avoid division by zero)
        group_means = group_sums / group_counts.clamp_min(1)

        # Map group means back to each sequence
        group_log_ratio_mean = group_means[group_indices]

        # K1 divergence: |group_mean(E[log(r)])|
        k1_div: torch.Tensor = group_log_ratio_mean.abs()
        log_ratio_for_metrics = k1_div.unsqueeze(-1)  # Store K1 divergence for metrics

        # For K1 divergence, threshold is max allowed divergence
        rs_statistic = k1_div.unsqueeze(-1).expand_as(log_ratio)

    elif rollout_rs == "group_k3":
        # Group-level masking with K3 KL estimator: reject entire groups of sequences together
        if group_indices is None:
            raise ValueError("group_indices must be provided when rollout_rs='group_k3'.")
        if torch.any(group_indices < 0):
            raise ValueError("`group_indices` must contain non-negative values.")

        # First compute sequence-level K3 (same as "k3" mode)
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        r = torch.exp(log_ratio_safe)  # Token-level ratio: r = exp(log_ratio_safe)
        k3_token = r - log_ratio_safe - 1  # K3 = r - log(r) - 1, using log(r) = log_ratio_safe
        k3_seq: torch.Tensor = verl_F.masked_mean(k3_token, response_mask, axis=-1)  # (batch_size,)

        # Vectorized group aggregation using scatter operations
        num_groups = group_indices.max().item() + 1
        device = k3_seq.device

        # Compute group sums and counts using scatter_add
        group_sums = torch.zeros(num_groups, device=device, dtype=k3_seq.dtype)
        group_counts = torch.zeros(num_groups, device=device, dtype=k3_seq.dtype)
        group_sums.scatter_add_(0, group_indices, k3_seq)
        group_counts.scatter_add_(0, group_indices, torch.ones_like(k3_seq))

        # Compute group means (avoid division by zero)
        group_means = group_sums / group_counts.clamp_min(1)

        # Map group means back to each sequence
        group_k3 = group_means[group_indices]

        log_ratio_for_metrics = group_k3.unsqueeze(-1)  # Store K3 values for metrics

        # For K3, use K3 divergence value for thresholding (NOT an IS ratio)
        rs_statistic = group_k3.unsqueeze(-1).expand_as(log_ratio)

    else:
        raise ValueError(f"Unsupported rollout_rs: {rollout_rs}")

    # Generate outlier mask based on mode
    # Divergence modes (k1, k3, group_k1, group_k3): divergence >= 0, reject if divergence > upper_threshold
    # Ratio modes (token, sequence, geometric): reject if outside [lower_threshold, upper_threshold]
    if rollout_rs in ["k1", "k3", "group_k1", "group_k3"]:
        # For divergence modes: divergence >= 0, reject if > upper_threshold
        # lower_threshold is ignored (divergence can't be negative)
        mask: torch.Tensor = rs_statistic <= upper_threshold
        mask = mask.float()
    else:
        # For ratio modes (token, sequence, geometric): reject if outside [lower_threshold, upper_threshold]
        mask = (rs_statistic >= lower_threshold) & (rs_statistic <= upper_threshold)
        mask = mask.float()

    # Compute rejection sampling metrics
    metrics: dict[str, float] = compute_rs_metrics(
        rs_statistic=rs_statistic,
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
    elif rollout_rs in ["group_k1", "group_k3"]:
        # Group-level: check fraction of groups rejected (and sequences)
        metrics["rollout_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()
        if group_indices is not None:
            # Vectorized: use scatter to check if any sequence in each group is rejected
            seq_rejected = 1 - mask[:, 0]  # 1 if rejected, 0 if kept
            num_groups_total = group_indices.max().item() + 1

            # Sum rejections per group (if any > 0, group has rejection)
            group_has_rejection = torch.zeros(num_groups_total, device=seq_rejected.device, dtype=seq_rejected.dtype)
            group_has_rejection.scatter_add_(0, group_indices, seq_rejected)

            # Count groups with at least one rejection
            groups_rejected = (group_has_rejection > 0).sum().item()
            num_groups_present = len(torch.unique(group_indices))
            metrics["rollout_rs_group_masked_fraction"] = (
                groups_rejected / num_groups_present if num_groups_present > 0 else 0.0
            )
    else:
        # Sequence-level aggregation: check first token's mask (all tokens in sequence have same mask)
        metrics["rollout_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()

    # Apply rejection mask to original response mask
    modified_response_mask: torch.Tensor = response_mask * mask

    return modified_response_mask, metrics


def compute_rs_metrics(
    rs_statistic: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str,
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
) -> dict[str, float]:
    """Compute metrics for hard trust region enforcement.

    This function calculates statistics for the trust region estimate used in
    masking, balancing numerical stability (using clamped values) and accuracy.

    Args:
        rs_statistic: Trust region statistic used for thresholding.
            - For ratio modes (token/sequence): IS ratio, ideal = 1.0
            - For divergence modes (geometric/k3/group_k1/group_k3): divergence, ideal = 0.0
            Shape: (batch_size, seq_length).
        log_ratio_for_metrics: Log ratio or divergence values for accurate metrics,
            shape varies by aggregation level.
        response_mask: Binary mask for valid tokens (1=valid, 0=padding),
            shape (batch_size, seq_length).
        rollout_rs: Trust region estimation level (matches compute_rollout_rejection_mask).
        rollout_rs_threshold: Trust region upper threshold.
        rollout_rs_threshold_lower: Trust region lower threshold (ignored for divergence modes).

    Returns:
        Dictionary of trust region metrics (all scalars).
    """
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device: torch.device = rs_statistic.device

    # Precompute log thresholds for accurate threshold checks
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold_lower, device=device))

    # Compute metrics based on aggregation level
    # Divergence modes: K1 (k1/group_k1) and K3 (k3/group_k3) output divergence >= 0
    # Ratio modes: token, sequence, and geometric output IS ratios
    if rollout_rs in ["k1", "group_k1"]:
        # K1 divergence modes: log_ratio_for_metrics contains |E[log(r)]| (>= 0)
        # K1 = |E[log(r)]|, threshold is max allowed K1 divergence
        k1_values = log_ratio_for_metrics  # Shape: (batch_size, 1)
        metrics["rollout_rs_k1_mean"] = k1_values.mean().item()
        metrics["rollout_rs_k1_max"] = k1_values.max().item()
        metrics["rollout_rs_k1_min"] = k1_values.min().item()
        # Note: rollout_rs_mean omitted for divergence modes to avoid confusion with IS ratio mean

        # Fraction exceeding threshold (K1 >= 0, so only upper threshold matters)
        exceeds_upper: torch.Tensor = k1_values > rollout_rs_threshold
        metrics["rollout_rs_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_rs_fraction_low"] = 0.0  # K1 divergence can't be negative

    elif rollout_rs in ["k3", "group_k3"]:
        # K3 divergence modes: log_ratio_for_metrics contains E[r - log(r) - 1] (>= 0)
        # K3 = E[r - log(r) - 1], threshold is max allowed K3 divergence
        k3_values = log_ratio_for_metrics  # Shape: (batch_size, 1)
        metrics["rollout_rs_k3_mean"] = k3_values.mean().item()
        metrics["rollout_rs_k3_max"] = k3_values.max().item()
        metrics["rollout_rs_k3_min"] = k3_values.min().item()
        # Note: rollout_rs_mean omitted for divergence modes to avoid confusion with IS ratio mean

        # Fraction exceeding threshold (K3 >= 0, so only upper threshold matters)
        exceeds_upper: torch.Tensor = k3_values > rollout_rs_threshold
        metrics["rollout_rs_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_rs_fraction_low"] = 0.0  # K3 divergence can't be negative

    elif rollout_rs in ["sequence", "geometric"]:
        # Sequence-level or geometric mean IS ratio: use log-space for accurate max/min/threshold checks
        # True max/min (unclamped) converted with safety bounds
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_rs_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_rs_min"] = torch.exp(log_min).item()

        # Mean uses clamped RS statistic to avoid overflow
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rs_statistic, response_mask).item()

        # Fraction exceeding thresholds (log-space for accuracy)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_rs_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_rs_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from clamped RS statistic
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rs_statistic, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rs_above_threshold: torch.Tensor = rs_statistic > rollout_rs_threshold
        rs_below_threshold: torch.Tensor = rs_statistic < rollout_rs_threshold_lower
        metrics["rollout_rs_fraction_high"] = verl_F.masked_mean(rs_above_threshold.float(), response_mask).item()
        metrics["rollout_rs_fraction_low"] = verl_F.masked_mean(rs_below_threshold.float(), response_mask).item()

        # Max/min (mask out padding tokens first)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_rs_max"] = rs_statistic.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_rs_min"] = rs_statistic.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped values for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        # Clamp to threshold range to avoid squaring extreme values
        # For divergence modes (k1, k3, group_k1, group_k3), lower bound is 0.0 (divergence >= 0)
        # For ratio modes, lower bound is rollout_rs_threshold_lower (reciprocal of upper)
        std_lower_bound = 0.0 if rollout_rs in ["k1", "k3", "group_k1", "group_k3"] else rollout_rs_threshold_lower
        stat_for_std: torch.Tensor = rs_statistic.clamp(min=std_lower_bound, max=rollout_rs_threshold)
        mean_clamped: torch.Tensor = verl_F.masked_mean(stat_for_std, response_mask)
        # Variance = E[X²] - (E[X])² (masked to valid tokens)
        rs_var: torch.Tensor = verl_F.masked_mean(stat_for_std.square(), response_mask) - mean_clamped.square()
        metrics["rollout_rs_std"] = torch.sqrt(torch.clamp(rs_var, min=0.0)).item()
    else:
        metrics["rollout_rs_std"] = 0.0

    # Compute Effective Sample Size (ESS) for RS statistic
    # ESS = 1 / E[(w_i / E[w_i])²] - only meaningful for ratio-based modes (not divergence modes)
    if rollout_rs in ["token", "sequence", "geometric"]:
        stat_for_ess: torch.Tensor = rs_statistic.clamp(min=rollout_rs_threshold_lower, max=rollout_rs_threshold)
        mean_for_ess: torch.Tensor = verl_F.masked_mean(stat_for_ess, response_mask)
        stat_normalized: torch.Tensor = stat_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
        metrics["rollout_rs_eff_sample_size"] = 1.0 / verl_F.masked_mean(stat_normalized.square(), response_mask).item()

    # Add sequence-level metrics if RS statistic has batch dimension
    if rs_statistic.dim() > 1:
        # Mean RS statistic per sequence (masked to valid tokens)
        seq_mean_stat: torch.Tensor = verl_F.masked_mean(rs_statistic, response_mask, axis=-1)

        metrics["rollout_rs_seq_mean"] = seq_mean_stat.mean().item()
        metrics["rollout_rs_seq_std"] = seq_mean_stat.std().item() if seq_mean_stat.numel() > 1 else 0.0
        metrics["rollout_rs_seq_max"] = seq_mean_stat.max().item()
        metrics["rollout_rs_seq_min"] = seq_mean_stat.min().item()

        # Sequence deviation from ideal value: 0.0 for divergence modes (k1, k3), 1.0 for ratio modes
        ideal_value = 0.0 if rollout_rs in ["k1", "k3", "group_k1", "group_k3"] else 1.0
        seq_deviation: torch.Tensor = (seq_mean_stat - ideal_value).abs()
        metrics["rollout_rs_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences exceeding thresholds
        metrics["rollout_rs_seq_fraction_high"] = (seq_mean_stat > rollout_rs_threshold).float().mean().item()
        metrics["rollout_rs_seq_fraction_low"] = (seq_mean_stat < rollout_rs_threshold_lower).float().mean().item()

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
        mask_float = response_mask.to(dtype=rollout_is_weights.dtype)
        if rollout_is == "token":
            # Token-level: normalize over all token weights
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(rollout_is_weights, mask_float)
            else:
                weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask)
        elif rollout_is == "sequence":
            # Sequence-level: normalize over sequence weights (one weight per sequence)
            # For each sequence, compute mean over valid tokens (they all have the same weight)
            # then average across sequences
            seq_weights = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)  # (batch_size,)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=rollout_is_weights.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(seq_weights, seq_mask)
            else:
                weights_mean = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)
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
    group_indices: Optional[torch.Tensor] = None,
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
            Options: None, "token", "sequence", "geometric", "k3", "group_k1", "group_k3".
            Set to None to disable rejection sampling.
        rollout_rs_threshold: Upper threshold for rejection sampling. Required if rollout_rs is enabled.
            For "k3" and "group_k3" modes, this is the max allowed K3 divergence (typical: 0.001-0.01).
            Default 2.0.
        rollout_rs_threshold_lower: Lower threshold for rejection sampling (used if rollout_rs is set).
            Defaults to 1/rollout_rs_threshold if None. Ignored for "k3" and "group_k3" modes.
        rollout_token_veto_threshold: Minimum allowed token-level IS weight. Sequences containing
            any token below this threshold are fully rejected. Set to None to disable veto.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch.
            Default: False.
        group_indices: Optional tensor of group indices, shape (batch_size,).
            Required when rollout_rs is "group_k1" or "group_k3". Sequences with the same
            index belong to the same group.

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
            group_indices=group_indices,
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

    # Get group_indices from batch if available (required for group_k1/group_k3 modes)
    group_indices = batch.batch.get("group_indices", None)

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
        group_indices=group_indices,
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


def apply_bypass_mode(
    batch: DataProto,
    rollout_corr_config: Optional[RolloutCorrectionConfig] = None,
    policy_loss_config: PolicyLossConfig = None,
) -> None:
    """
    Setup bypass mode: Use rollout_log_probs as old_log_probs.

    Bypass mode skips expensive actor forward pass for old_log_prob computation
    by setting old_log_probs = rollout_log_probs (2 policies instead of 3).

    Uses compute_policy_loss_bypass_mode() which supports:
    - loss_type="ppo_clip" (default): PPO clipped objective (IS handled by ratio)
    - loss_type="reinforce": REINFORCE with explicit IS weights

    Both loss types benefit from rejection sampling (RS) which masks out-of-distribution samples.

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
        # Pass rollout_correction config to actor for loss computation and metrics
        policy_loss_config["rollout_correction"] = rollout_corr_config
        # Always use bypass_mode loss function which handles both loss_types
        policy_loss_config["loss_mode"] = "bypass_mode"
