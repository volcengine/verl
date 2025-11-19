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
    "When Speed Kills Stability: Demystifying RL Collapse from the Inference-Training Mismatch"
    https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

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
            - "token": Reject individual tokens with outlier ratios
            - "sequence": Reject entire sequences with outlier ratios
            - "geometric": Geometric mean aggregation (threshold: 1.0002-1.001)
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
            - If ANY token has ratio < threshold, entire sequence is rejected
            - Independent of rollout_is and rollout_rs settings
            - Typical values: 1e-4 to 1e-6 when enabled
            Default: None (disabled)

        bypass_mode (bool): Operating mode - bypass or decoupled.
            - True: Bypass mode - reuse rollout_log_prob as old_log_prob (2 policies)
            - False: Decoupled mode - compute old_log_prob separately (3 policies)
            Default: False (decoupled mode)

        use_policy_gradient (bool): Loss function type.
            - Requires bypass_mode=True
            - True: Policy gradient loss (no PPO clipping)
            - False: PPO loss (with clipping)
            Default: False (PPO loss)

        rollout_is_batch_normalize (bool): Apply batch normalization to IS weights.
            - True: Normalize IS weights to have mean=1.0 within each batch
            - False: Use raw (truncated) IS weights (standard)
            - Reduces variance by ensuring average weight is 1.0 per batch
            - Only affects IS weight values, not rejection sampling
            Default: False (no batch normalization)

    Example:
        # Create with defaults
        config = RolloutCorrectionConfig()

        # Use presets
        config = RolloutCorrectionConfig.decoupled_token_is()  # Decoupled mode with token-level IS
        config = RolloutCorrectionConfig.decoupled_seq_is_rs()  # Decoupled mode with sequence IS + RS
        config = RolloutCorrectionConfig.decoupled_seq_is()  # Decoupled mode with sequence-level IS
        config = RolloutCorrectionConfig.ppo_is_bypass()  # Bypass mode
        config = RolloutCorrectionConfig.pg_is()  # Policy gradient with IS
        config = RolloutCorrectionConfig.pg_rs()  # Policy gradient with RS

    Reference:
        Liu, Li, Fu, Wang, Liu, Shen (2025)
        "When Speed Kills Stability: Demystifying RL Collapse from the Inference-Training Mismatch"
        https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
    """

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[float] = None
    rollout_rs_threshold_lower: Optional[float] = None
    rollout_token_veto_threshold: Optional[float] = None
    bypass_mode: bool = False
    use_policy_gradient: bool = False
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
    def decoupled_geo_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Geometric Rejection Sampling.

        Uses geometric mean for rejection sampling at sequence level in decoupled mode,
        with additional veto mechanism. Geometric mean is extremely sensitive to outliers,
        requiring very tight thresholds close to 1.0.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with geometric RS + veto
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def ppo_is_bypass(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """PPO with IS Correction in Bypass Mode.

        Skips old_log_prob computation by reusing rollout_log_prob.
        PPO clips against rollout policy instead of true old policy.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for PPO_IS bypass mode
        """
        return cls(
            rollout_is="token",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            use_policy_gradient=False,
        )

    @classmethod
    def pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Policy Gradient with IS Correction.

        Uses policy gradient loss with explicit IS correction.
        No PPO clipping.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for PG with IS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            use_policy_gradient=True,
        )

    @classmethod
    def pg_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Policy Gradient with Rejection Sampling.

        Policy gradient with rejection sampling (no IS weights) using geometric mean in bypass mode.
        Skips old_log_prob computation for faster execution.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for PG with RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            use_policy_gradient=True,
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
            - RolloutCorrectionConfig.decoupled_geo_rs() - Decoupled mode with geometric RS + veto
            - RolloutCorrectionConfig.ppo_is_bypass() - Bypass mode (skips old_log_prob)
            - RolloutCorrectionConfig.pg_is() - Policy gradient with IS
            - RolloutCorrectionConfig.pg_rs() - Policy gradient with RS

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
