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

from dataclasses import dataclass
from typing import Optional


@dataclass
class KLControlConfig:
    """Configuration for KL control."""
    type: str = "fixed"  # "fixed" or "adaptive"
    kl_coef: float = 0.001  # Initial coefficient for KL penalty
    horizon: int = 10000  # Horizon value for adaptive controller
    target_kl: float = 0.1  # Target KL divergence for adaptive controller


@dataclass
class PFPPOConfig:
    """Configuration for preference feedback PPO."""
    reweight_method: str = "pow"  # "pow", "max_min", or "max_random"
    weight_pow: float = 2.0  # Power used for weight scaling in "pow" method


@dataclass
class AlgorithmConfig:
    """Configuration for the algorithm."""
    gamma: float = 1.0  # Discount factor for future rewards
    lam: float = 1.0  # Trade-off between bias and variance in the GAE estimator
    adv_estimator: str = "gae"  # Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
    norm_adv_by_std_in_grpo: bool = True  # Whether to normalize advantages by std (specific to GRPO)
    use_kl_in_reward: bool = False  # Whether to enable in-reward KL penalty
    kl_penalty: str = "kl"  # How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full"
    kl_ctrl: Optional[KLControlConfig] = None  # KL control configuration
    use_pf_ppo: bool = False  # Whether to enable preference feedback PPO
    pf_ppo: Optional[PFPPOConfig] = None  # Preference feedback PPO settings

    def __post_init__(self):
        """Initialize nested configs if they are None."""
        if self.kl_ctrl is None:
            self.kl_ctrl = KLControlConfig()
        if self.pf_ppo is None:
            self.pf_ppo = PFPPOConfig()

    def get(self, key: str, default=None):
        """Get attribute with default value for backward compatibility."""
        return getattr(self, key, default)