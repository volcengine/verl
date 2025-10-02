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

"""FlowRL Advantage Estimator Implementation.

This module provides the FlowRL advantage estimator that can be used
as a drop-in replacement for other estimators like GRPO.
"""

import torch
from typing import Optional
import numpy as np

from verl.trainer.config import AlgoConfig


def compute_flowrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for FlowRL using trajectory balance.

    This is a simplified version that treats FlowRL rewards similar to GRPO
    but prepares the data structure for the FlowRL actor loss computation.

    The actual trajectory balance loss is computed in the FlowRL actor,
    but this estimator normalizes the advantages properly.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping responses by prompt
        epsilon: `(float)`
            small value to avoid division by zero
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    # Compute sequence-level scores (sum of token rewards)
    scores = token_level_rewards.sum(dim=-1)

    # Group-based normalization similar to GRPO
    # This ensures advantages are centered per prompt
    from collections import defaultdict
    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Collect scores by prompt group
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        # Compute mean per group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # Center the scores (subtract group mean)
        # For FlowRL, we don't divide by std like GRPO
        # because the trajectory balance loss handles scaling
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        # Broadcast scalar advantage across token dimension
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# Register the FlowRL advantage estimator
def register_flowrl_estimator():
    """Register FlowRL advantage estimator with the core registry."""
    try:
        from verl.trainer.ppo.core_algos import register_adv_est

        # Register as 'flowrl'
        register_adv_est("flowrl")(compute_flowrl_outcome_advantage)

        print("FlowRL advantage estimator registered successfully as 'flowrl'")
        return True
    except ImportError as e:
        print(f"Warning: Could not register FlowRL estimator: {e}")
        return False
    except Exception as e:
        print(f"Error registering FlowRL estimator: {e}")
        return False


# Auto-register when module is imported
register_flowrl_estimator()
