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
RARO-specific utility functions for dual-pass rollout and joint loss computation.

This module provides utilities for:
1. Dual-pass rollout (Policy generation -> Critic evaluation)
2. Joint GRPO loss computation with weighted Policy and Critic objectives
3. Batch mixing with replay buffer
"""

import random
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from verl import DataProto
from verl.workers.reward_manager.raro_prompts import (
    RAROPrompts,
    compute_raro_rewards,
    parse_critic_label,
)
from verl.workers.reward_manager.raro_replay_buffer import RAROReplayBuffer, ReservoirBuffer


@dataclass
class RARORolloutOutput:
    """Output from the dual-pass RARO rollout.

    Attributes:
        policy_data: Data from Policy rollout pass
        critic_data: Data from Critic rollout pass
        policy_rewards: Rewards for Policy role
        critic_rewards: Rewards for Critic role
        outcomes: Outcome types ('correct', 'deceived', 'tie', 'parse_error')
        critic_labels: Parsed labels from Critic outputs
    """

    policy_data: DataProto
    critic_data: Optional[DataProto] = None
    policy_rewards: Optional[np.ndarray] = None
    critic_rewards: Optional[np.ndarray] = None
    outcomes: Optional[np.ndarray] = None
    critic_labels: Optional[np.ndarray] = None


def create_critic_batch_from_policy_output(
    policy_data: DataProto,
    replay_buffer: RAROReplayBuffer | ReservoirBuffer,
    replay_buffer_ratio: float = 0.5,
    shuffle_answers: bool = True,
    prompts: Optional[RAROPrompts] = None,
    tokenizer: Optional[Any] = None,
) -> tuple[list[dict[str, Any]], list[Literal["Answer 1", "Answer 2"]]]:
    """Create Critic batch from Policy output with replay buffer mixing.

    Args:
        policy_data: DataProto from Policy rollout
        replay_buffer: Replay buffer for historical samples
        replay_buffer_ratio: Ratio of historical to fresh samples
        shuffle_answers: Whether to shuffle answer order
        prompts: Prompt templates (default: None, uses DEFAULT_PROMPTS)
        tokenizer: Tokenizer for decoding text

    Returns:
        Tuple of (critic_batch, expert_positions)
        where critic_batch is a list of dicts with 'question', 'answer_1', 'answer_2'
        and expert_positions indicates which position contains the expert answer
    """
    prompts = prompts or RAROPrompts()
    critic_batch = []
    expert_positions = []

    # Extract current generations
    current_generations = []
    for i, data_item in enumerate(policy_data):
        # Decode prompt and response
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        attention_mask = data_item.batch["attention_mask"]
        valid_prompt_length = attention_mask[:prompt_length].sum().item()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = attention_mask[prompt_length:].sum().item()
        valid_response_ids = response_ids[:valid_response_length]

        if tokenizer is not None:
            prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Remove EOS token if present
            eos_token = tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
        else:
            prompt_str = ""
            response_str = ""

        # Get expert answer from extra_info
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        expert_answer = extra_info.get("answer", "")
        question = _extract_question_from_prompt(prompt_str)

        current_generations.append({
            "question_id": data_item.non_tensor_batch.get("uid", i),
            "question": question,
            "expert_answer": expert_answer,
            "policy_answer": response_str,
        })

    # Sample from replay buffer
    n_samples = len(current_generations)
    n_historical = int(n_samples * replay_buffer_ratio)
    n_fresh = n_samples - n_historical

    historical_samples = replay_buffer.sample(min(n_historical, len(replay_buffer)))

    # Use all current generations (or sample if too many)
    if len(current_generations) > n_fresh:
        fresh_samples = random.sample(current_generations, n_fresh)
    else:
        fresh_samples = current_generations

    # Combine and create critic prompts
    all_samples = fresh_samples + historical_samples

    for sample in all_samples:
        if shuffle_answers and random.random() < 0.5:
            # Shuffle: Answer 1 is expert, Answer 2 is policy
            critic_batch.append({
                "question": sample["question"],
                "answer_1": sample["expert_answer"],
                "answer_2": sample["policy_answer"],
            })
            expert_positions.append("Answer 1")
        else:
            # No shuffle: Answer 1 is expert, Answer 2 is policy
            critic_batch.append({
                "question": sample["question"],
                "answer_1": sample["expert_answer"],
                "answer_2": sample["policy_answer"],
            })
            expert_positions.append("Answer 1")

    return critic_batch, expert_positions


def _extract_question_from_prompt(prompt: str) -> str:
    """Extract the question from the prompt.

    Args:
        prompt: The full prompt string

    Returns:
        The extracted question
    """
    import re

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

    return prompt.strip()


def compute_joint_raro_loss(
    policy_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    lambda_pol: float = 1.0 / 9.0,
    lambda_crit: float = 8.0 / 9.0,
) -> torch.Tensor:
    """Compute the weighted joint loss for RARO.

    The joint objective is:
    J(θ) = λ_pol * J_pol(θ) + λ_crit * J_crit(θ)

    Args:
        policy_loss: Policy role loss
        critic_loss: Critic role loss
        lambda_pol: Weight for Policy loss (default: 1/9)
        lambda_crit: Weight for Critic loss (default: 8/9)

    Returns:
        Combined loss tensor
    """
    # Normalize weights to sum to 1
    total_weight = lambda_pol + lambda_crit
    normalized_lambda_pol = lambda_pol / total_weight
    normalized_lambda_crit = lambda_crit / total_weight

    joint_loss = normalized_lambda_pol * policy_loss + normalized_lambda_crit * critic_loss
    return joint_loss


def compute_raro_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """Compute the clipped PPO/GRPO loss for the Policy role.

    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage estimates
        response_mask: Mask for response tokens
        clip_range: PPO clipping parameter

    Returns:
        Policy loss tensor
    """
    # Compute ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Compute clipped surrogate loss
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    # Take minimum and apply mask
    policy_loss = torch.max(pg_losses, pg_losses2)
    policy_loss = masked_mean(policy_loss, response_mask)

    return policy_loss


def compute_raro_critic_loss(
    critic_log_probs: torch.Tensor,
    critic_old_log_probs: torch.Tensor,
    critic_rewards: torch.Tensor,
    critic_mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """Compute the loss for the Critic role.

    The Critic is trained to correctly identify the expert answer using
    a classification-style loss based on the reward signal.

    Args:
        critic_log_probs: Current Critic log probabilities
        critic_old_log_probs: Old Critic log probabilities
        critic_rewards: Rewards for Critic (1.0 for correct, 0.0 for deceived, tau for tie)
        critic_mask: Mask for Critic tokens
        clip_range: PPO clipping parameter

    Returns:
        Critic loss tensor
    """
    # For Critic, we use the reward as a pseudo-label
    # Higher reward = better performance = preserve behavior
    # We use a policy gradient style loss for the Critic

    ratio = torch.exp(critic_log_probs - critic_old_log_probs)

    # Negative because we want to maximize reward
    pg_losses = -critic_rewards * ratio
    pg_losses2 = -critic_rewards * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    critic_loss = torch.max(pg_losses, pg_losses2)
    critic_loss = masked_mean(critic_loss, critic_mask)

    return critic_loss


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Compute the mean of a tensor with a mask.

    Args:
        tensor: Input tensor
        mask: Boolean mask tensor
        dim: Dimension(s) to reduce over (default: all)

    Returns:
        Masked mean tensor
    """
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    else:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


def aggregate_raro_metrics(
    policy_rewards: np.ndarray,
    critic_rewards: np.ndarray,
    outcomes: np.ndarray,
    critic_labels: np.ndarray,
) -> dict[str, float]:
    """Aggregate metrics from RARO training.

    Args:
        policy_rewards: Array of Policy rewards
        critic_rewards: Array of Critic rewards
        outcomes: Array of outcome types
        critic_labels: Array of Critic labels

    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {}

    # Policy metrics
    metrics["raro/policy_reward_mean"] = float(np.mean(policy_rewards))
    metrics["raro/policy_reward_std"] = float(np.std(policy_rewards))
    metrics["raro/policy_success_rate"] = float(np.mean(policy_rewards == 1.0))  # Deception rate

    # Critic metrics
    metrics["raro/critic_reward_mean"] = float(np.mean(critic_rewards))
    metrics["raro/critic_reward_std"] = float(np.std(critic_rewards))
    metrics["raro/critic_accuracy"] = float(np.mean(critic_rewards == 1.0))  # Correct identification

    # Outcome distribution
    unique_outcomes, counts = np.unique(outcomes, return_counts=True)
    for outcome, count in zip(unique_outcomes, counts):
        metrics[f"raro/outcome_{outcome}_rate"] = float(count / len(outcomes))

    # Label distribution
    unique_labels, label_counts = np.unique(critic_labels, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        if label != "Unknown":
            metrics[f"raro/label_{label.lower()}_rate"] = float(count / len(critic_labels))

    # Tie rate (important metric from paper)
    tie_rate = float(np.sum(outcomes == "tie") / len(outcomes))
    metrics["raro/tie_rate"] = tie_rate

    return metrics


class RARODualPassRollout:
    """Wrapper for implementing dual-pass rollout in RARO.

    This class manages the two-stage process:
    1. Policy Rollout: Generate answers for questions
    2. Critic Rollout: Judge which answer is expert vs policy
    """

    def __init__(
        self,
        replay_buffer: RAROReplayBuffer | ReservoirBuffer,
        prompts: Optional[RAROPrompts] = None,
        replay_buffer_ratio: float = 0.5,
        shuffle_answers: bool = True,
        tau_pol: float = 0.6,
        tau_crit: float = 0.55,
    ):
        """Initialize the dual-pass rollout manager.

        Args:
            replay_buffer: Buffer for storing historical generations
            prompts: Prompt templates (default: None, uses DEFAULT_PROMPTS)
            replay_buffer_ratio: Ratio of historical to fresh samples
            shuffle_answers: Whether to shuffle answer order for Critic
            tau_pol: Policy reward weight for Tie outcome
            tau_crit: Critic reward weight for Tie outcome
        """
        self.replay_buffer = replay_buffer
        self.prompts = prompts or RAROPrompts()
        self.replay_buffer_ratio = replay_buffer_ratio
        self.shuffle_answers = shuffle_answers
        self.tau_pol = tau_pol
        self.tau_crit = tau_crit

    def execute_policy_pass(self, data: DataProto, tokenizer) -> RARORolloutOutput:
        """Execute the Policy rollout pass.

        Args:
            data: Input DataProto with questions
            tokenizer: Tokenizer for decoding

        Returns:
            RARORolloutOutput with Policy data
        """
        # Store generations in replay buffer
        for i, data_item in enumerate(data):
            # Decode and store (similar to _store_policy_generations)
            # ... implementation ...

        return RARORolloutOutput(policy_data=data)

    def execute_critic_pass(
        self,
        policy_output: RARORolloutOutput,
        critic_model,
        tokenizer,
    ) -> RARORolloutOutput:
        """Execute the Critic rollout pass.

        Args:
            policy_output: Output from Policy pass
            critic_model: Model for Critic evaluation
            tokenizer: Tokenizer for decoding

        Returns:
            RARORolloutOutput with Critic data and rewards
        """
        # Create Critic batch
        critic_batch, expert_positions = create_critic_batch_from_policy_output(
            policy_data=policy_output.policy_data,
            replay_buffer=self.replay_buffer,
            replay_buffer_ratio=self.replay_buffer_ratio,
            shuffle_answers=self.shuffle_answers,
            prompts=self.prompts,
            tokenizer=tokenizer,
        )

        # Run Critic model (this would be done by the actual rollout worker)
        # ... implementation ...

        # Compute rewards
        # ... implementation ...

        return policy_output

    def compute_final_rewards(
        self,
        critic_outputs: list[str],
        expert_positions: list[Literal["Answer 1", "Answer 2"]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute rewards from Critic outputs.

        Args:
            critic_outputs: List of Critic judgment outputs
            expert_positions: List indicating which position had expert answer

        Returns:
            Tuple of (policy_rewards, critic_rewards, outcomes, labels)
        """
        n = len(critic_outputs)
        policy_rewards = np.zeros(n)
        critic_rewards = np.zeros(n)
        outcomes = np.empty(n, dtype=object)
        labels = np.empty(n, dtype=object)

        for i, (output, expert_pos) in enumerate(zip(critic_outputs, expert_positions)):
            label = parse_critic_label(output)
            labels[i] = label

            pol_r, crit_r, outcome = compute_raro_rewards(
                critic_label=label,
                expert_position=expert_pos,
                tau_pol=self.tau_pol,
                tau_crit=self.tau_crit,
            )

            policy_rewards[i] = pol_r
            critic_rewards[i] = crit_r
            outcomes[i] = outcome

        return policy_rewards, critic_rewards, outcomes, labels
