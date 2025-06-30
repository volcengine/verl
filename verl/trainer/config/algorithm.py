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
from typing import Optional

from verl.base_config import BaseConfig


@dataclass(frozen=True)
class KLControlConfig(BaseConfig):
    """Configuration for KL control."""

    type: str = "fixed"  # "fixed" or "adaptive"
    kl_coef: float = 0.001  # Initial coefficient for KL penalty
    horizon: int = 10000  # Horizon value for adaptive controller
    target_kl: float = 0.1  # Target KL divergence for adaptive controller


@dataclass(frozen=True)
class PFPPOConfig(BaseConfig):
    """Configuration for preference feedback PPO."""

    reweight_method: str = "pow"  # "pow", "max_min", or "max_random"
    weight_pow: float = 2.0  # Power used for weight scaling in "pow" method


@dataclass(frozen=True)
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy)."""

    enable: bool = False  # Whether to enable filter groups
    metric: Optional[str] = None  # Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
    max_num_gen_batches: int = 0  # Non-positive values mean no upper limit


@dataclass(frozen=True)
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm."""

    gamma: float = 1.0  # Discount factor for future rewards
    lam: float = 1.0  # Trade-off between bias and variance in the GAE estimator
    adv_estimator: str = "gae"  # Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
    norm_adv_by_std_in_grpo: bool = True  # Whether to normalize advantages by std (specific to GRPO)
    use_kl_in_reward: bool = False  # Whether to enable in-reward KL penalty
    kl_penalty: str = "kl"  # How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)  # KL control configuration
    use_pf_ppo: bool = False  # Whether to enable preference feedback PPO
    pf_ppo: Optional[PFPPOConfig] = None  # Preference feedback PPO settings

    # Filter groups parameters (used in DAPO and Entropy)
    filter_groups: Optional[FilterGroupsConfig] = None  # Filter groups configuration
