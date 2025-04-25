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
The base class for reward model
"""

from abc import ABC, abstractmethod
import torch

from verl import DataProto
from verl.utils.length_penalty import apply_length_penalty


class BasePPORewardModel(ABC):
    def __init__(self, config):
        self.config = config
        # Initialize length penalty parameters
        self.length_penalty_config = config.reward_model.get("length_penalty", {})
        self.use_length_penalty = self.length_penalty_config.get("enabled", False)
        self.length_penalty_alpha = self.length_penalty_config.get("alpha", 0.0)
        self.length_penalty_min_length = self.length_penalty_config.get("min_length", 0)
        self.length_penalty_max_length = self.length_penalty_config.get("max_length", None)

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        """Computing reward given input_ids. The transformers should output a tensor with shape
           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: a data pass protocol containing "reward". Only the [EOS] position contains the reward.
            Other position should have zero reward. Note that this may change in the future if we use
            dense reward. So, we leave the interface for general case.
            - reward: [batch_size, sequence_length].

        """
        pass

    def apply_length_penalty(self, rewards, sequence_lengths):
        """
        Apply length penalty to the rewards if enabled.
        
        Args:
            rewards: Tensor of shape [batch_size] containing rewards
            sequence_lengths: Tensor of shape [batch_size] containing sequence lengths
            
        Returns:
            Tensor of shape [batch_size] with potentially modified rewards
        """
        if not self.use_length_penalty:
            return rewards
            
        return apply_length_penalty(
            rewards,
            sequence_lengths,
            alpha=self.length_penalty_alpha,
            min_length=self.length_penalty_min_length,
            max_length=self.length_penalty_max_length
        )

    def _get_sequence_lengths(self, data):
        """
        Extract response lengths from the data.
        
        Args:
            data: DataProto object with input_ids and attention_mask
            
        Returns:
            Tensor of shape [batch_size] containing sequence lengths
        """
        if "response_lengths" in data:
            return data["response_lengths"]
        
        # If response_lengths not provided, try to compute from attention_mask
        if "attention_mask" in data and "prompt_lengths" in data:
            # Calculate response length by subtracting prompt length from total length
            total_lengths = torch.sum(data["attention_mask"], dim=1)
            prompt_lengths = data["prompt_lengths"]
            return total_lengths - prompt_lengths
            
        return None