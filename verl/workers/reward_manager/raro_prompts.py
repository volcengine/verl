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
Prompt templates for RARO (Relativistic Adversarial Reasoning Optimization)

Defines the system prompts for Policy and Critic roles in the adversarial game.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RAROPrompts:
    """Prompt templates for RARO adversarial training.

    Attributes:
        policy_system_prompt: System prompt for the Policy role (generates answers)
        critic_system_prompt: System prompt for the Critic role (judges answers)
    """

    # Policy System Prompt - standard assistant role
    policy_system_prompt: str = (
        "You are a helpful assistant with strong reasoning capabilities. "
        "When presented with a question, provide a clear, step-by-step solution "
        "followed by your final answer. Format your final answer using \\boxed{{}} notation."
    )

    # Critic System Prompt - adversarial judge role
    critic_system_prompt: str = (
        "You are an expert critic trained to distinguish between expert solutions and "
        "attempted solutions. Your task is to carefully compare two answers and determine "
        "which one is more likely to be the expert answer, or if they are indistinguishable.\n\n"
        "Analyze both answers thoroughly, examining:\n"
        "- Logical correctness and coherence\n"
        "- Mathematical accuracy\n"
        "- Clarity and precision of reasoning\n"
        "- Proper notation and formatting\n\n"
        "After your analysis, output EXACTLY ONE of the following labels:\n"
        "[Answer 1] - if Answer 1 is clearly the expert answer\n"
        "[Answer 2] - if Answer 2 is clearly the expert answer\n"
        "[Tie] - if both answers are of similar quality or cannot be distinguished"
    )

    # Template for formatting the comparison task
    critic_comparison_template: str = (
        "## Question\n{question}\n\n"
        "## Answer 1\n{answer_1}\n\n"
        "## Answer 2\n{answer_2}\n\n"
        "## Instructions\n"
        "Compare the two answers above. Determine which one is the expert answer, "
        "or if they are indistinguishable. Provide your reasoning, then output exactly "
        "one of: [Answer 1], [Answer 2], or [Tie]"
    )

    def get_policy_prompt(self, question: str) -> str:
        """Get the full prompt for Policy generation.

        Args:
            question: The input question

        Returns:
            Formatted prompt for Policy role
        """
        return f"{self.policy_system_prompt}\n\n## Question\n{question}\n\nProvide your solution:"

    def get_critic_prompt(
        self, question: str, answer_1: str, answer_2: str, shuffle_answers: bool = True
    ) -> tuple[str, Literal["Answer 1", "Answer 2"]]:
        """Get the full prompt for Critic comparison.

        Args:
            question: The input question
            answer_1: First answer to compare
            answer_2: Second answer to compare
            shuffle_answers: Whether to shuffle answer order for robustness

        Returns:
            Tuple of (formatted_prompt, expert_position) where expert_position
            indicates which position actually contains the expert answer
        """
        if shuffle_answers:
            import random

            if random.random() < 0.5:
                # Answer 1 is expert, Answer 2 is policy
                prompt = self.critic_comparison_template.format(
                    question=question, answer_1=answer_1, answer_2=answer_2
                )
                expert_position = "Answer 1"
            else:
                # Answer 2 is expert, Answer 1 is policy
                prompt = self.critic_comparison_template.format(
                    question=question, answer_1=answer_2, answer_2=answer_1
                )
                expert_position = "Answer 2"
        else:
            # No shuffling - Answer 1 is expert, Answer 2 is policy
            prompt = self.critic_comparison_template.format(
                question=question, answer_1=answer_1, answer_2=answer_2
            )
            expert_position = "Answer 1"

        return f"{self.critic_system_prompt}\n\n{prompt}", expert_position


# Default instance
DEFAULT_PROMPTS = RAROPrompts()


# Parse critic output to extract label
def parse_critic_label(critic_output: str) -> str:
    """Parse the critic's output to extract the judgment label.

    Args:
        critic_output: The raw text output from the Critic

    Returns:
        One of: 'Answer 1', 'Answer 2', 'Tie', or 'Unknown' if parsing fails
    """
    output_lower = critic_output.lower()

    # Look for the label in various formats
    if "[answer 1]" in output_lower or "answer 1" in output_lower:
        return "Answer 1"
    elif "[answer 2]" in output_lower or "answer 2" in output_lower:
        return "Answer 2"
    elif "[tie]" in output_lower or "tie" in output_lower:
        return "Tie"
    else:
        return "Unknown"


# Compute rewards based on the reward matrix
def compute_raro_rewards(
    critic_label: str,
    expert_position: Literal["Answer 1", "Answer 2"],
    tau_pol: float = 0.6,
    tau_crit: float = 0.55,
) -> tuple[float, float, str]:
    """Compute rewards for Policy and Critic based on their adversarial outcome.

    Args:
        critic_label: The label output by the Critic ('Answer 1', 'Answer 2', 'Tie', or 'Unknown')
        expert_position: Which position actually contained the expert answer
        tau_pol: Reward weight for Policy when outcome is Tie
        tau_crit: Reward weight for Critic when outcome is Tie

    Returns:
        Tuple of (policy_reward, critic_reward, outcome_type)
        where outcome_type is one of: 'correct', 'deceived', 'tie', 'parse_error'
    """
    # Handle parsing errors
    if critic_label == "Unknown":
        return 0.0, 0.0, "parse_error"

    # Determine if Critic correctly identified the expert answer
    critic_correct = (critic_label == expert_position)

    # Determine outcome type
    if critic_label == "Tie":
        # Tie outcome
        return tau_pol, tau_crit, "tie"
    elif critic_correct:
        # Critic correctly identified expert
        return 0.0, 1.0, "correct"
    else:
        # Critic was deceived (identified policy as expert)
        return 1.0, 0.0, "deceived"
