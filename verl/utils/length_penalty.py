# Copyright 2025 Individual Contributors
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

import torch

def compute_length_penalty(sequence_lengths, alpha=0.0, min_length=0, max_length=None):
    """
    Compute length penalty for sequence generation.
    
    Args:
        sequence_lengths: Tensor of shape [batch_size] containing sequence lengths
        alpha: Float controlling the strength and direction of the penalty:
               - alpha > 0: Favor longer sequences
               - alpha < 0: Favor shorter sequences
               - alpha = 0: No length penalty
        min_length: Minimum sequence length before applying penalty (no penalty below this)
        max_length: Maximum sequence length (sequences longer than this get max penalty)
    
    Returns:
        Tensor of shape [batch_size] containing length penalties to be applied to rewards
    """
    if alpha == 0.0:
        return torch.ones_like(sequence_lengths, dtype=torch.float32)
    
    effective_lengths = sequence_lengths.clone().float()
    
    if min_length > 0:
        effective_lengths = torch.maximum(effective_lengths - min_length, 
                                         torch.zeros_like(effective_lengths))
    
    if max_length is not None:
        effective_lengths = torch.minimum(effective_lengths, 
                                         torch.ones_like(effective_lengths) * (max_length - min_length))
    
    # Calculate penalty using the standard formula: ((5 + length)/6)^alpha
    # This is similar to the formula used in Google Neural Machine Translation (GNMT) paper
    penalty = ((5.0 + effective_lengths) / 6.0) ** alpha
    
    return penalty

def apply_length_penalty(rewards, sequence_lengths, **kwargs):
    """
    Apply length penalty to rewards.
    
    Args:
        rewards: Tensor of shape [batch_size] containing the original rewards
        sequence_lengths: Tensor of shape [batch_size] containing sequence lengths
        **kwargs: Parameters for compute_length_penalty
    
    Returns:
        Tensor of shape [batch_size] with length-penalized rewards
    """
    length_penalties = compute_length_penalty(sequence_lengths, **kwargs)
    return rewards * length_penalties