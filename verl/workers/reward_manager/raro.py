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

"""
RARO (Relativistic Adversarial Reasoning Optimization) Reward Manager

Implements the adversarial game between Policy and Critic as described in the
"Escaping the Verifier: Learning to Reason via Demonstrations" paper.
"""

import re
from collections import defaultdict
from typing import Any, Literal, Optional

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from .raro_prompts import DEFAULT_PROMPTS, RAROPrompts, compute_raro_rewards, parse_critic_label
from .raro_replay_buffer import RAROReplayBuffer, ReservoirBuffer


@register("raro")
class RARORewardManager(AbstractRewardManager):
    """Reward Manager for RARO adversarial training.

    This manager implements a dual-pass rollout system:
    1. Policy Rollout: Generate policy answers for questions
    2. Critic Rollout: Judge which answer is expert vs policy

    The rewards are computed based on the adversarial outcome:
    - Policy gets 1.0 if it deceives the Critic, 0.0 otherwise, tau_pol for Tie
    - Critic gets 1.0 if correctly identifies expert, 0.0 if deceived, tau_crit for Tie

    Attributes:
        tokenizer: Tokenizer for decoding/encoding text
        num_examine: Number of samples to examine for debugging
        tau_pol: Policy reward weight for Tie outcome
        tau_crit: Critic reward weight for Tie outcome
        replay_buffer_size: Maximum size of the replay buffer
        replay_buffer_ratio: Ratio of historical to fresh samples in Critic batch
        max_response_length: Maximum allowed response length
        shuffle_answers: Whether to shuffle answer order for Critic
        prompts: Custom prompt templates (optional)
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 1,
        compute_score=None,
        reward_fn_key: str = "data_source",
        tau_pol: float = 0.6,
        tau_crit: float = 0.55,
        replay_buffer_size: int = 10000,
        replay_buffer_ratio: float = 0.5,
        max_response_length: int = 4096,
        shuffle_answers: bool = True,
        buffer_type: Literal["fifo", "reservoir"] = "fifo",
        prompts: Optional[RAROPrompts] = None,
        **kwargs,
    ) -> None:
        """Initialize the RARO reward manager.

        Args:
            tokenizer: Tokenizer for decoding/encoding text
            num_examine: Number of samples to examine for debugging
            compute_score: Unused (kept for compatibility)
            reward_fn_key: Unused (kept for compatibility)
            tau_pol: Policy reward weight for Tie outcome (default: 0.6)
            tau_crit: Critic reward weight for Tie outcome (default: 0.55)
            replay_buffer_size: Maximum size of the replay buffer (default: 10000)
            replay_buffer_ratio: Ratio of historical to fresh samples (default: 0.5)
            max_response_length: Maximum allowed response length (default: 4096)
            shuffle_answers: Whether to shuffle answer order (default: True)
            buffer_type: Type of buffer - 'fifo' or 'reservoir' (default: 'fifo')
            prompts: Custom prompt templates (default: None, uses DEFAULT_PROMPTS)
            **kwargs: Additional arguments
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.tau_pol = tau_pol
        self.tau_crit = tau_crit
        self.replay_buffer_ratio = replay_buffer_ratio
        self.max_response_length = max_response_length
        self.shuffle_answers = shuffle_answers
        self.prompts = prompts or DEFAULT_PROMPTS

        # Initialize replay buffer
        if buffer_type == "reservoir":
            self.replay_buffer = ReservoirBuffer(capacity=replay_buffer_size)
        else:
            self.replay_buffer = RAROReplayBuffer(capacity=replay_buffer_size)

        # Statistics tracking
        self.stats = defaultdict(int)

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Process a batch and compute rewards based on adversarial outcome.

        This function handles the dual-pass rollout:
        1. Decode policy generations from the data
        2. Extract expert answers from extra_info
        3. For each sample, determine if Critic was already run
        4. If Critic output exists, parse and compute rewards
        5. Otherwise, this is a Policy-only pass (rewards will be computed later)

        Args:
            data: DataProto containing the batch
            return_dict: Whether to return additional info

        Returns:
            Tensor of rewards or dict with rewards and extra info
        """
        # Check if we have pre-computed rewards from Critic pass
        if "raro_rewards" in data.non_tensor_batch:
            # Extract pre-computed rewards
            raro_rewards = data.non_tensor_batch["raro_rewards"]
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

            for i, reward in enumerate(raro_rewards):
                # Get valid response length
                response_ids = data.batch["responses"][i]
                attention_mask = data.batch["attention_mask"][i]
                prompt_length = data.batch["prompts"][i].shape[-1]
                valid_response_length = attention_mask[prompt_length:].sum().item()

                # Place reward at the last valid token
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = reward

            if return_dict:
                reward_extra_info = {
                    "critic_labels": data.non_tensor_batch.get("critic_labels", np.array(["Unknown"] * len(data))),
                    "outcomes": data.non_tensor_batch.get("outcomes", np.array(["parse_error"] * len(data))),
                }
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                return reward_tensor

        # Policy-only pass: rewards will be computed after Critic rollout
        # Return zero rewards for now (this happens during initial generation)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Store policy generations in replay buffer for future Critic training
        self._store_policy_generations(data)

        if return_dict:
            reward_extra_info = {
                "note": "Policy pass only - rewards computed after Critic rollout",
            }
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    def _store_policy_generations(self, data: DataProto) -> None:
        """Store policy generations in the replay buffer.

        Args:
            data: DataProto containing the batch with policy generations
        """
        for i, data_item in enumerate(data):
            # Get prompt
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch["attention_mask"]
            valid_prompt_length = attention_mask[:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Get response
            response_ids = data_item.batch["responses"]
            valid_response_length = attention_mask[prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Remove EOS token if present
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # Get expert answer and question from extra_info
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            expert_answer = extra_info.get("answer", "")
            question_id = data_item.non_tensor_batch.get("uid", i)

            # Extract question from prompt (assuming format after "<|User|>" or similar)
            question = self._extract_question(prompt_str)

            # Store in replay buffer
            self.replay_buffer.add(
                question_id=question_id,
                question=question,
                expert_answer=expert_answer,
                policy_answer=response_str,
                metadata={"prompt": prompt_str},
            )

    def _extract_question(self, prompt: str) -> str:
        """Extract the question from the prompt.

        Args:
            prompt: The full prompt string

        Returns:
            The extracted question
        """
        # Try common patterns for question extraction
        patterns = [
            r"<\|User\|>\s*(.*?)\s*<\|Assistant\|>",
            r"<｜User｜>\s*(.*?)\s*<｜Assistant｜>",
            r"User:\s*(.*?)\s*Assistant:",
            r"## Question\n(.*?)\n",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: return the last part of prompt
        return prompt.strip()

    def create_critic_batch(
        self, current_generations: list[dict[str, Any]], n_samples: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Create a mixed batch for Critic training.

        Combines fresh generations with samples from the replay buffer
        to prevent catastrophic forgetting.

        Args:
            current_generations: List of current (q, a_exp, a_pol) triplets
            n_samples: Target batch size (default: len(current_generations))

        Returns:
            Mixed list of triplets for Critic evaluation
        """
        if n_samples is None:
            n_samples = len(current_generations)

        # Determine split between fresh and historical
        n_fresh = int(n_samples * (1 - self.replay_buffer_ratio))
        n_historical = n_samples - n_fresh

        # Sample from current generations
        if len(current_generations) <= n_fresh:
            fresh_samples = current_generations
        else:
            import random

            fresh_samples = random.sample(current_generations, n_fresh)

        # Sample from replay buffer
        historical_samples = self.replay_buffer.sample(n_historical)

        # Combine
        critic_batch = fresh_samples + historical_samples

        return critic_batch

    def compute_critic_rewards(
        self,
        question: str,
        expert_answer: str,
        policy_answer: str,
        critic_output: str,
        expert_position: Literal["Answer 1", "Answer 2"],
    ) -> tuple[float, float, str]:
        """Compute rewards for a single (q, a_exp, a_pol, critic_output) tuple.

        Args:
            question: The input question
            expert_answer: The expert/expert answer
            policy_answer: The policy-generated answer
            critic_output: The Critic's judgment output
            expert_position: Which position contained the expert answer

        Returns:
            Tuple of (policy_reward, critic_reward, outcome_type)
        """
        # Parse critic label
        critic_label = parse_critic_label(critic_output)

        # Compute rewards using the reward matrix
        policy_reward, critic_reward, outcome = compute_raro_rewards(
            critic_label=critic_label,
            expert_position=expert_position,
            tau_pol=self.tau_pol,
            tau_crit=self.tau_crit,
        )

        # Update statistics
        self.stats[f"outcome_{outcome}"] += 1
        self.stats[f"label_{critic_label}"] += 1

        return policy_reward, critic_reward, outcome

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics.

        Returns:
            Dictionary with statistics
        """
        buffer_stats = self.replay_buffer.get_statistics() if hasattr(self.replay_buffer, "get_statistics") else {}

        return {
            **dict(self.stats),
            **buffer_stats,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats.clear()


def create_raro_batch_with_critic_outputs(
    data: DataProto,
    critic_outputs: list[str],
    expert_positions: list[Literal["Answer 1", "Answer 2"]],
    rewards: list[tuple[float, float, str]],
    tokenizer,
) -> DataProto:
    """Create a new DataProto with Critic outputs and rewards.

    This is a helper function for creating the batch after Critic evaluation.

    Args:
        data: Original DataProto from Policy rollout
        critic_outputs: List of Critic outputs
        expert_positions: List of expert positions for each sample
        rewards: List of (policy_reward, critic_reward, outcome) tuples
        tokenizer: Tokenizer for encoding

    Returns:
        New DataProto with Critic information added
    """
    n = len(data)

    # Create arrays for additional data
    raro_rewards = np.array([r[0] for r in rewards], dtype=object)
    critic_labels = np.array([parse_critic_label(output) for output in critic_outputs], dtype=object)
    outcomes = np.array([r[2] for r in rewards], dtype=object)
    expert_positions_array = np.array(expert_positions, dtype=object)

    # Create non_tensor_batch dict
    non_tensor_batch = dict(data.non_tensor_batch)
    non_tensor_batch["raro_rewards"] = raro_rewards
    non_tensor_batch["critic_outputs"] = np.array(critic_outputs, dtype=object)
    non_tensor_batch["critic_labels"] = critic_labels
    non_tensor_batch["outcomes"] = outcomes
    non_tensor_batch["expert_positions"] = expert_positions_array

    # Create new DataProto
    return DataProto(
        batch=data.batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=data.meta_info,
    )
