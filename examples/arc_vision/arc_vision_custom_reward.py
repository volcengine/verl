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
"""Custom reward function for Arc Vision RL training."""

import json
import logging
from typing import Dict, List, Any

import torch

from verl import DataProto
from verl.utils.reward_score.arc_vision_reward import ArcVisionRewardScore

logger = logging.getLogger(__name__)


def arc_vision_compute_reward(data: DataProto, 
                            return_dict: bool = False,
                            confidence_threshold: float = 0.7,
                            reward_weights: Dict[str, float] = None,
                            tool_penalties: Dict[str, float] = None,
                            **kwargs):
    """Custom reward function for Arc Vision that integrates with VERL's PPO trainer.
    
    This function is called by VERL's reward manager to compute rewards for
    Arc Vision responses that include tool usage for UI element detection.
    
    Args:
        data: DataProto containing prompts and responses
        confidence_threshold: Threshold for confidence-gated tool invocation
        reward_weights: Weights for reward components (task, tool, gate)
        tool_penalties: Penalties for different tool usage failure modes
        **kwargs: Additional keyword arguments
        
    Returns:
        torch.Tensor: Reward scores for each response in the batch
    """
    # Initialize Arc Vision reward model
    reward_model = ArcVisionRewardScore(
        confidence_threshold=confidence_threshold,
        reward_weights=reward_weights,
        tool_penalties=tool_penalties
    )
    
    rewards = []
    
    # Process each item in the batch
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem
        
        # Extract prompt and response strings
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # Decode to strings (tokenizer passed via kwargs)
        tokenizer = kwargs.get("tokenizer")
        if tokenizer:
            prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        else:
            # Fallback to raw strings if available
            prompt_str = data_item.non_tensor_batch.get("raw_prompt", "")
            response_str = data_item.non_tensor_batch.get("response", "")
        
        # Extract ground truth from reward_model dict
        reward_model_data = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model_data.get("ground_truth", None)
        
        if ground_truth is None:
            # No ground truth available for this sample
            rewards.append(0.0)
            continue
        
        # Prepare ground truth data structure
        gt_data = {"ground_truth": ground_truth}
        
        # Compute reward using Arc Vision reward model
        reward_list = reward_model(
            questions=[prompt_str],
            responses=[response_str],
            reward_model=[gt_data]
        )
        rewards.append(reward_list[0])
    
    # Convert to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # Log some statistics for debugging
    if len(rewards) > 0:
        logger.info(f"Arc Vision rewards - Mean: {reward_tensor.mean():.3f}, "
                   f"Std: {reward_tensor.std():.3f}, "
                   f"Min: {reward_tensor.min():.3f}, "
                   f"Max: {reward_tensor.max():.3f}")
    
    if return_dict:
        return {"reward_tensor": reward_tensor}
    else:
        return reward_tensor