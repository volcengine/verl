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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "RolloutCorrectionConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Configuration for Rollout Correction (addresses off-policy issues in RL training).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Rollout Correction handles off-policiness from multiple sources:
    1. Policy mismatch: Rollout policy (e.g., vLLM BF16) vs Training policy (e.g., FSDP FP32)
    2. Model update staleness: Rollout data collected from older policy checkpoints
    3. General off-policy scenarios: Any distribution shift between data collection and training

    For more details, see:
    "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
    https://richardli.xyz/rl-collapse

    This typed config replaces the old dict-based approach and provides:
    - Type safety and validation
    - Clear documentation of all parameters
    - Named factory methods for common presets (TIS, MIS, etc.)
    - Sensible defaults

    Args:
        rollout_is (Optional[str]): IS weight aggregation level.
            - None: No IS weights (metrics only)
            - "token": Per-token IS weights (low variance, biased)
            - "sequence": Per-sequence IS weights (unbiased, high variance)
            Default: "sequence"

        rollout_is_threshold (float): Upper threshold for IS weight truncation/rejection.
            Typical range: 1.5-5.0 for token level, 2.0-10.0 for sequence level.
            Default: 2.0

        rollout_rs (Optional[str]): Rejection sampling aggregation level.
            - None: No rejection sampling
            - "token": Reject individual tokens with outlier ratios (ideal=1.0)
            - "sequence": Reject entire sequences with outlier ratios (ideal=1.0)
            - "geometric": Geometric mean IS ratio exp(E[log(r)]) (ideal=1.0)
              Length-normalized ratio, uses [lower, upper] threshold bounds.
              (typical: 1.0002-1.001)
            - "k1": K1 KL divergence at sequence level: |E[log(r)]|
              K1 >= 0, so only upper threshold applies (typical: 0.0002-0.001).
            - "k3": K3 KL estimator at sequence level: E[r - log(r) - 1]
              More stable than K1 for small KL values. K3 >= 0, so only
              upper threshold applies (typical: 0.001-0.01).
            For details about "k{n}" KL estimators, see http://joschu.net/blog/kl-approx.html
            Default: None (use IS weights without rejection)

        rollout_rs_threshold (Optional[float]): Upper threshold for rejection sampling.
            - If None and rollout_rs is enabled, uses rollout_is_threshold
            - Tokens/sequences with ratio > threshold are masked out
            Default: None (uses rollout_is_threshold when rollout_rs is enabled)

        rollout_rs_threshold_lower (Optional[float]): Lower threshold for rejection sampling.
            - If None, uses reciprocal of upper threshold (1/upper)
            - Tokens/sequences with ratio < threshold are masked out
            Default: None (auto-computed as reciprocal)

        rollout_token_veto_threshold (Optional[float]): Per-token veto for catastrophic outliers.
            - Checks unclamped per-token ratios before safety bounds
            - If ANY token has abs(π_train/π_rollout) > threshold, the entire sequence is rejected
            - Threshold must be > 1.0; values ≤ 1.0 veto every sequence
            - Independent of rollout_is and rollout_rs settings
            - Typical values: 1e2-1e4 when enabled (tune per use case)
            Default: None (disabled)

        bypass_mode (bool): Operating mode - bypass or decoupled.
            - True: Bypass mode - reuse rollout_log_prob as old_log_prob (2 policies)
              Uses compute_policy_loss_bypass_mode() with loss_type selection
            - False: Decoupled mode - compute old_log_prob separately (3 policies)
              Uses standard PPO loss with IS weight correction
            Default: False (decoupled mode)

        loss_type (str): Loss function type in bypass mode (bypass_mode=True).
            - "reinforce": REINFORCE-style policy gradient with explicit IS weights
              L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
            - "ppo_clip": PPO clipped objective (IS handled by ratio, no explicit weights)
              L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Default: "ppo_clip"

        rollout_is_batch_normalize (bool): Apply batch normalization to IS weights.
            - True: Normalize IS weights to have mean=1.0 within each batch
            - False: Use raw (truncated) IS weights (standard)
            - Reduces variance by ensuring average weight is 1.0 per batch
            - Only affects IS weight values, not rejection sampling
            Default: False (no batch normalization)

    Example:
        # Create with defaults
        config = RolloutCorrectionConfig()

        # Decoupled PPO mode presets (3 policies: π_rollout, π_old, π_θ)
        # IS weights correct for gap between π_old and π_rollout
        config = RolloutCorrectionConfig.decoupled_token_is()  # Token-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is()    # Seq-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is_rs() # Seq-MIS
        config = RolloutCorrectionConfig.decoupled_k1_rs()     # K1-RS (divergence mode)
        config = RolloutCorrectionConfig.decoupled_geo_rs()    # Geo-RS (ratio mode)

        # Bypass mode presets (2 policies: π_rollout = π_old, π_θ)
        # loss_type controls the loss function
        # PPO-clip presets (ratio handles IS, so no separate IS weights needed):
        config = RolloutCorrectionConfig.bypass_ppo_clip()              # PPO-clip only
        config = RolloutCorrectionConfig.bypass_ppo_clip_k1_rs()        # PPO-clip + K1-RS
        config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()       # PPO-clip + Geo-RS
        config = RolloutCorrectionConfig.bypass_ppo_clip_k3_rs()        # PPO-clip + K3-RS
        # REINFORCE presets (explicit IS weights):
        config = RolloutCorrectionConfig.bypass_pg_is()                 # REINFORCE + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_k1_rs()              # REINFORCE + K1-RS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs()             # REINFORCE + Geo-RS
        config = RolloutCorrectionConfig.bypass_pg_k1_rs_seq_tis()      # REINFORCE + K1-RS + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_k1_rs_token_tis()    # REINFORCE + K1-RS + Token-TIS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_seq_tis()     # REINFORCE + Geo-RS + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_token_tis()   # REINFORCE + Geo-RS + Token-TIS

        # Decoupled K1 divergence presets (length-invariant, solves Length Trap)
        config = RolloutCorrectionConfig.decoupled_k1_rs_seq_tis()            # Decoupled K1-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_k1_rs_token_tis()          # Decoupled K1-RS + Token-TIS

        # Decoupled Geometric ratio presets (length-normalized IS ratio)
        config = RolloutCorrectionConfig.decoupled_geo_rs_seq_tis()           # Decoupled Geo-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_geo_rs_token_tis()         # Decoupled Geo-RS + Token-TIS

        # Decoupled K3 KL Estimator presets (more stable for small KL values)
        config = RolloutCorrectionConfig.decoupled_k3_rs()                    # Decoupled K3-RS
        config = RolloutCorrectionConfig.decoupled_k3_rs_seq_tis()            # Decoupled K3-RS + Seq-TIS
        config = RolloutCorrectionConfig.decoupled_k3_rs_token_tis()          # Decoupled K3-RS + Token-TIS

    Reference:
        Liu, Li, Fu, Wang, Liu, Shen (2025)
        "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
        https://richardli.xyz/rl-collapse
    """

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[float] = None
    rollout_rs_threshold_lower: Optional[float] = None
    rollout_token_veto_threshold: Optional[float] = None
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"
    rollout_is_batch_normalize: bool = False

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Token-level Importance Sampling.

        IS weight correction at token level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with token-level IS
        """
        return cls(rollout_is="token", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level Importance Sampling.

        IS weight correction at sequence level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence-level IS
        """
        return cls(rollout_is="sequence", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is_rs(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 2.0,
        rs_threshold_lower: Optional[float] = None,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level IS + Rejection Sampling.

        Sequence-level IS with sequence-level rejection sampling in decoupled mode.
        Rejects entire sequences based on sequence-level IS weight.

        Args:
            is_threshold (float): Upper threshold for IS weights. Default: 2.0
            rs_threshold (float): Upper threshold for rejection sampling. Default: 2.0
            rs_threshold_lower (Optional[float]): Lower threshold for rejection sampling.
                If None, auto-computed as reciprocal of rs_threshold. Default: None

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence IS + RS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="sequence",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
        )

    @classmethod
    def decoupled_k1_rs(
        cls,
        rs_threshold: float = 0.001,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with K1 Rejection Sampling.

        Uses K1 divergence |E[log(r)]| for rejection sampling at sequence level in decoupled mode,
        with additional veto mechanism. K1 divergence is sensitive to policy changes.

        Args:
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
                Rejects sequences where |E[log(π_train/π_rollout)]| > threshold.
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with K1-RS + veto
        """
        return cls(
            rollout_is=None,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def decoupled_geo_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Geometric Mean Rejection Sampling (ratio-based).

        Uses geometric mean IS ratio exp(E[log(r)]) for rejection sampling at sequence level.
        This is a ratio-based mode (ideal = 1.0) with [lower, upper] threshold bounds.
        Length-normalized but still uses IS ratio semantics.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with Geo-RS + veto
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def bypass_ppo_clip(cls) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss.

        PPO clipped objective in bypass mode. The PPO ratio = π_θ/π_rollout
        already handles IS correction, so no explicit IS weights are applied.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip
        """
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_k1_rs(
        cls,
        rs_threshold: float = 0.001,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and K1 Rejection Sampling.

        PPO clipped objective in bypass mode with K1 divergence RS to mask outliers.
        The PPO ratio = π_θ/π_rollout already handles IS correction.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).
        Solves the "Length Trap" problem for CoT/agent workloads.

        Args:
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
                Rejects sequences where |E[log(π_train/π_rollout)]| > threshold.
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + K1-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_geo_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and Geometric Mean RS (ratio-based).

        PPO clipped objective in bypass mode with geometric mean IS ratio RS.
        Uses exp(E[log(r)]) (ideal = 1.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_k3_rs(
        cls,
        rs_threshold: float = 0.01,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and K3 Rejection Sampling.

        PPO clipped objective in bypass mode with K3 KL estimator RS to mask outliers.
        K3 is more stable than K1 for small KL values.
        The PPO ratio = π_θ/π_rollout already handles IS correction.

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + K3-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="k3",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and IS Correction.

        Uses REINFORCE loss with explicit IS correction in bypass mode.
        No PPO clipping.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + IS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_k1_rs(
        cls,
        rs_threshold: float = 0.001,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and K1 Rejection Sampling.

        REINFORCE with K1 divergence rejection sampling (no IS weights) in bypass mode.
        Skips old_log_prob computation for faster execution.

        Solves the "Length Trap" problem where standard IS estimators penalize long sequences.
        Suitable for reasoning models (CoT) and agents with long action sequences.

        Args:
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
                Rejects sequences where |E[log(π_train/π_rollout)]| > threshold.
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + K1-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and Geometric Mean RS (ratio-based).

        REINFORCE with geometric mean IS ratio rejection sampling in bypass mode.
        Uses exp(E[log(r)]) (ideal = 1.0) with [lower, upper] threshold bounds.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_k1_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.001,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K1 RS and Sequence-level Truncated IS (K1-RS-Seq-TIS).

        Combines the K1 divergence filter (length-invariant validity check) with
        Clipped Sequence Weight (debiasing).

        Suitable for reasoning models (CoT, o1-style) and agents that need to
        think for many steps without collapsing.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for K1-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def decoupled_k1_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.001,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K1 RS and Token-level Truncated IS (K1-RS-Token-TIS).

        Combines the K1 divergence filter (length-invariant validity check) with
        Token-level IS weights (lower variance, biased).

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for K1-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def decoupled_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Sequence-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Clipped Sequence Weight (debiasing). Uses exp(E[log(r)]) (ideal = 1.0).

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower). Default: None
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def decoupled_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with Geometric Mean RS and Token-level Truncated IS (ratio-based).

        Combines the Geometric Mean Filter (ratio-based validity check) with
        Token-level IS weights. Uses exp(E[log(r)]) (ideal = 1.0).

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower). Default: None
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def bypass_pg_k1_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.001,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, K1-RS, and Sequence-level IS.

        Combines K1 divergence rejection with sequence-level IS
        in bypass mode with REINFORCE loss (no PPO clipping).

        Suitable for reasoning models (CoT, o1-style) and agents when you want
        bypass mode efficiency.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + K1-RS + Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_k1_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.001,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, K1-RS, and Token-level IS.

        Combines K1 divergence rejection with token-level IS weights
        in bypass mode with REINFORCE loss (no PPO clipping).

        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): K1 divergence threshold (max allowed). Default: 0.001 (±0.1%)
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + K1-RS + Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="k1",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Sequence-level IS.

        Combines geometric mean IS ratio rejection with sequence-level IS
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses exp(E[log(r)]) (ideal = 1.0) with [lower, upper] threshold bounds.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower). Default: None
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Token-level IS.

        Combines geometric mean IS ratio rejection with token-level IS weights
        in bypass mode with REINFORCE loss (no PPO clipping).
        Uses exp(E[log(r)]) (ideal = 1.0) with [lower, upper] threshold bounds.

        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower). Default: None
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_k3_rs(
        cls,
        rs_threshold: float = 0.01,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 KL Estimator Rejection Sampling.

        Uses K3 KL estimator at sequence level for rejection sampling.
        K3 = E[r - log(r) - 1] where r = π_train/π_rollout.
        More stable than geometric mean for small KL values.

        K3 >= 0 always (equals 0 when policies match exactly).

        Args:
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
                Typical range: 0.001-0.1
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for K3 RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="k3",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def decoupled_k3_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Sequence-level Truncated IS.

        Combines K3 KL estimator rejection with sequence-level IS weights.
        K3 provides more stable outlier detection than geometric mean.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="k3",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def decoupled_k3_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled mode with K3 RS and Token-level Truncated IS.

        Combines K3 KL estimator rejection with token-level IS weights.
        K3 provides more stable outlier detection than geometric mean.
        Token-level IS has lower variance but introduces bias.

        Args:
            is_threshold (float): Upper threshold for token IS weights. Default: 2.0
            rs_threshold (float): Max allowed K3 divergence. Default: 0.01
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for K3-RS-Token-TIS
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="k3",
            rollout_rs_threshold=rs_threshold,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        """Disabled - Metrics Only Mode.

        Computes and logs off-policy metrics without applying correction.

        Returns:
            RolloutCorrectionConfig with all correction disabled
        """
        return cls(rollout_is=None, rollout_rs=None)


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_correction (Optional[RolloutCorrectionConfig]): Rollout Correction configuration.
            Addresses off-policy issues from policy mismatch, model staleness, and general distribution shifts.

            Set to None to disable entirely. Use factory methods for common presets:
            - RolloutCorrectionConfig.decoupled_token_is() - Decoupled mode with token-level IS
            - RolloutCorrectionConfig.decoupled_seq_is() - Decoupled mode with sequence-level IS
            - RolloutCorrectionConfig.decoupled_seq_is_rs() - Decoupled mode with sequence IS + RS
            - RolloutCorrectionConfig.decoupled_k1_rs() - Decoupled mode with K1-RS (divergence)
            - RolloutCorrectionConfig.decoupled_geo_rs() - Decoupled mode with Geo-RS (ratio)
            - RolloutCorrectionConfig.bypass_ppo_clip() - Bypass mode with PPO-clip
            - RolloutCorrectionConfig.bypass_ppo_clip_k1_rs() - Bypass mode with PPO-clip + K1-RS
            - RolloutCorrectionConfig.bypass_pg_is() - Bypass mode with REINFORCE + IS
            - RolloutCorrectionConfig.bypass_pg_k1_rs() - Bypass mode with REINFORCE + K1-RS

            For backward compatibility, you can still pass a dict, which will be converted to
            RolloutCorrectionConfig automatically.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Correction: corrects off-policy issues (policy mismatch, model staleness, distribution shifts)
    # Set to None to disable, use RolloutCorrectionConfig presets (e.g., .tis(), .mis()), or pass dict
    rollout_correction: Optional[RolloutCorrectionConfig] = None
