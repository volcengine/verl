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
import numpy as np

def compute_length_penalty(sequence_lengths, alpha=0.0, min_length=0, max_length=None):
    """
    Compute length penalty for sequence generation.
    
    Args:
        sequence_lengths: Tensor or numpy array of shape [batch_size] containing sequence lengths
        alpha: Float controlling the strength and direction of the penalty:
               - alpha > 0: Favor longer sequences
               - alpha < 0: Favor shorter sequences
               - alpha = 0: No length penalty
        min_length: Minimum sequence length before applying penalty (no penalty below this)
        max_length: Maximum sequence length (sequences longer than this get max penalty)
    
    Returns:
        Tensor or numpy array of shape [batch_size] containing length penalties
    """
    if alpha == 0.0:
        if isinstance(sequence_lengths, torch.Tensor):
            return torch.ones_like(sequence_lengths, dtype=torch.float32)
        else:
            return np.ones_like(sequence_lengths, dtype=np.float32)
    
    effective_lengths = sequence_lengths.copy() if isinstance(sequence_lengths, np.ndarray) else sequence_lengths.clone()
    
    if isinstance(effective_lengths, np.ndarray):
        effective_lengths = effective_lengths.astype(np.float32)
    else:
        effective_lengths = effective_lengths.float()
    
    if min_length > 0:
        if isinstance(effective_lengths, np.ndarray):
            effective_lengths = np.maximum(effective_lengths - min_length, np.zeros_like(effective_lengths))
        else:
            effective_lengths = torch.maximum(effective_lengths - min_length, 
                                             torch.zeros_like(effective_lengths))
    
    if max_length is not None:
        if isinstance(effective_lengths, np.ndarray):
            effective_lengths = np.minimum(effective_lengths, np.ones_like(effective_lengths) * (max_length - min_length))
        else:
            effective_lengths = torch.minimum(effective_lengths, 
                                            torch.ones_like(effective_lengths) * (max_length - min_length))
    
    # Calculate penalty using the standard formula: ((5 + length)/6)^alpha
    # This is similar to the formula used in Google Neural Machine Translation paper [https://arxiv.org/pdf/1609.08144]
    if isinstance(effective_lengths, np.ndarray):
        penalty = ((5.0 + effective_lengths) / 6.0) ** alpha
    else:
        penalty = ((5.0 + effective_lengths) / 6.0) ** alpha
    
    return penalty

def apply_length_penalty(rewards, sequence_lengths, **kwargs):
    """
    Apply length penalty to rewards.
    
    Args:
        rewards: List, Tensor or numpy array of shape [batch_size] containing the original rewards
        sequence_lengths: List, Tensor or numpy array of shape [batch_size] containing sequence lengths
        **kwargs: Parameters for compute_length_penalty
    
    Returns:
        List, Tensor or numpy array of shape [batch_size] with length-penalized rewards
    """
    import numpy as np
    import torch
    
    is_tensor = isinstance(rewards, torch.Tensor)
    is_list = isinstance(rewards, list)
    
    if is_list:
        rewards_np = np.array(rewards)
    elif is_tensor:
        rewards_np = rewards.cpu().numpy()
    else:
        rewards_np = rewards
        
    if isinstance(sequence_lengths, torch.Tensor):
        sequence_lengths_np = sequence_lengths.cpu().numpy()
    elif isinstance(sequence_lengths, list):
        sequence_lengths_np = np.array(sequence_lengths)
    else:
        sequence_lengths_np = sequence_lengths
    
    length_penalties = compute_length_penalty(sequence_lengths_np, **kwargs)
    
    penalized_rewards_np = rewards_np * length_penalties
    
    if is_tensor:
        device = rewards.device
        penalized_rewards = torch.tensor(penalized_rewards_np, dtype=rewards.dtype, device=device)
    elif is_list:
        penalized_rewards = penalized_rewards_np.tolist()
    else:
        penalized_rewards = penalized_rewards_np
    
    return penalized_rewards