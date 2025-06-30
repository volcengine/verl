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
    Pad sequences to same length.
    
    Args:
        sequences: List of tensors to pad
        pad_value: Value to use for padding
        max_length: Maximum length (if None, use longest sequence)
        
    Returns:
        torch.Tensor: Padded sequences
    """
    if not sequences:
        return torch.empty(0)
    
    # Find max length
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    
    # Pad sequences
    padded = []
    for seq in sequences:
        if seq.shape[0] < max_length:
            padding = torch.full(
                (max_length - seq.shape[0],) + seq.shape[1:],
                pad_value,
                dtype=seq.dtype,
                device=seq.device
            )
            seq = torch.cat([seq, padding], dim=0)
        elif seq.shape[0] > max_length:
            seq = seq[:max_length]
        padded.append(seq)
    
    return torch.stack(padded)


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