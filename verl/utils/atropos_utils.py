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

"""
Utility functions for Atropos integration.

This module provides utility functions for data validation, conversion,
and GPU operations for the Atropos-VeRL integration.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def validate_api_response(response: Dict[str, Any]) -> bool:
    """
    Validate Atropos API response format.
    
    Args:
        response: Response dictionary from Atropos API
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ["advantages", "metrics"]
    
    if not isinstance(response, dict):
        logger.error("Response is not a dictionary")
        return False
    
    for field in required_fields:
        if field not in response:
            logger.error(f"Missing required field: {field}")
            return False
    
    return True


def convert_advantages_to_tensor(
    advantages: Union[List[float], np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert advantages to PyTorch tensor.
    
    Args:
        advantages: Advantages in various formats
        device: Target device for tensor
        dtype: Data type for tensor
        
    Returns:
        torch.Tensor: Converted advantages
    """
    if isinstance(advantages, torch.Tensor):
        tensor = advantages
    elif isinstance(advantages, np.ndarray):
        tensor = torch.from_numpy(advantages)
    else:
        tensor = torch.tensor(advantages)
    
    tensor = tensor.to(dtype=dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def pad_token_sequences(
    sequences: List[torch.Tensor],
    pad_value: float = 0.0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Pad sequences to same length using efficient padding.
    
    Args:
        sequences: List of tensors to pad
        pad_value: Value to use for padding
        max_length: Maximum length (if None, use longest sequence)
        
    Returns:
        torch.Tensor: Padded sequences
    """
    if not sequences:
        return torch.empty(0)
    
    # Truncate sequences if max_length is specified
    if max_length is not None:
        sequences = [seq[:max_length] for seq in sequences]
    
    # Use torch.nn.utils.rnn.pad_sequence for efficient padding
    # Note: pad_sequence expects (sequence_length, *) tensors and returns (max_len, batch_size, *)
    # We need to transpose to get (batch_size, max_len, *)
    from torch.nn.utils.rnn import pad_sequence
    
    # Ensure all sequences have same number of dimensions
    if sequences and len(sequences[0].shape) == 1:
        # 1D sequences - pad_sequence handles this directly
        padded = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    else:
        # Multi-dimensional sequences - flatten extra dims temporarily
        original_shape = sequences[0].shape[1:]
        flat_sequences = [seq.view(seq.shape[0], -1) for seq in sequences]
        padded_flat = pad_sequence(flat_sequences, batch_first=True, padding_value=pad_value)
        # Restore original shape
        padded = padded_flat.view(padded_flat.shape[0], padded_flat.shape[1], *original_shape)
    
    return padded


def compute_batch_metrics(advantages: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for a batch of advantages.
    
    Args:
        advantages: Advantage tensor
        mask: Mask tensor for valid tokens
        
    Returns:
        Dict[str, float]: Computed metrics
    """
    if mask.sum() == 0:
        return {
            "mean_advantage": 0.0,
            "std_advantage": 0.0,
            "min_advantage": 0.0,
            "max_advantage": 0.0,
            "valid_tokens": 0
        }
    
    valid_advantages = advantages[mask.bool()]
    
    return {
        "mean_advantage": valid_advantages.mean().item(),
        "std_advantage": valid_advantages.std().item(),
        "min_advantage": valid_advantages.min().item(),
        "max_advantage": valid_advantages.max().item(),
        "valid_tokens": mask.sum().item()
    }


def merge_batch_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Merge metrics from multiple batches.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dict[str, float]: Merged metrics
    """
    if not metrics_list:
        return {}
    
    total_tokens = sum(m.get("valid_tokens", 0) for m in metrics_list)
    
    if total_tokens == 0:
        return {
            "mean_advantage": 0.0,
            "std_advantage": 0.0,
            "min_advantage": 0.0,
            "max_advantage": 0.0,
            "total_tokens": 0
        }
    
    # Weighted average for mean
    mean_advantage = sum(
        m.get("mean_advantage", 0.0) * m.get("valid_tokens", 0)
        for m in metrics_list
    ) / total_tokens
    
    # Global min/max
    min_advantage = min(m.get("min_advantage", float('inf')) for m in metrics_list)
    max_advantage = max(m.get("max_advantage", float('-inf')) for m in metrics_list)
    
    return {
        "mean_advantage": mean_advantage,
        "min_advantage": min_advantage,
        "max_advantage": max_advantage,
        "total_tokens": total_tokens
    }