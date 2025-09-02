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
Trajectory reward computation for multi-turn GRPO.
This module implements reward computation logic for complete trajectories.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer

from verl.workers.rollout.multi_turn_trajectory import MultiTurnTrajectory, TrajectoryStep


class TrajectoryRewardComputer(ABC):
    """Abstract base class for computing trajectory-level rewards."""
    
    @abstractmethod
    def compute_reward(self, trajectory: MultiTurnTrajectory) -> float:
        """Compute reward for a complete trajectory.
        
        Args:
            trajectory: Complete multi-turn trajectory
            
        Returns:
            Reward value for the trajectory
        """
        pass


class GRPOTrajectoryRewardComputer(TrajectoryRewardComputer):
    """
    GRPO-style trajectory reward computer.
    
    Reward logic:
    - 0 for bad format
    - 0 for wrong answer
    - reward = 1 / max_context_length otherwise
    """
    
    def __init__(self, 
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 ground_truth_answers: Optional[Dict[str, Any]] = None,
                 format_validators: Optional[List[callable]] = None,
                 answer_validators: Optional[List[callable]] = None):
        """Initialize the reward computer.
        
        Args:
            tokenizer: Tokenizer for computing context lengths
            ground_truth_answers: Dictionary mapping question IDs to ground truth answers
            format_validators: List of functions to validate trajectory format
            answer_validators: List of functions to validate answer correctness
        """
        self.tokenizer = tokenizer
        self.ground_truth_answers = ground_truth_answers or {}
        self.format_validators = format_validators or [self._default_format_validator]
        self.answer_validators = answer_validators or [self._default_answer_validator]
    
    def compute_reward(self, trajectory: MultiTurnTrajectory) -> float:
        """Compute GRPO-style reward for trajectory.
        
        Args:
            trajectory: Complete multi-turn trajectory
            
        Returns:
            Reward value: 0 for bad format/wrong answer, 1/max_context_length otherwise
        """
        # Check format validity
        if not self._is_valid_format(trajectory):
            trajectory.is_valid_format = False
            return 0.0
        
        # Check answer correctness
        if not self._is_correct_answer(trajectory):
            trajectory.is_correct_answer = False
            return 0.0
        
        # Compute context-length-based reward
        max_context_length = self._compute_max_context_length(trajectory)
        trajectory.max_context_length = max_context_length
        trajectory.is_valid_format = True
        trajectory.is_correct_answer = True
        
        return 1.0 / max(max_context_length, 1)  # Avoid division by zero
    
    def _is_valid_format(self, trajectory: MultiTurnTrajectory) -> bool:
        """Check if trajectory has valid format.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            True if format is valid
        """
        for validator in self.format_validators:
            if not validator(trajectory):
                return False
        return True
    
    def _is_correct_answer(self, trajectory: MultiTurnTrajectory) -> bool:
        """Check if trajectory produces correct answer.
        
        Args:
            trajectory: Trajectory to evaluate
            
        Returns:
            True if answer is correct
        """
        for validator in self.answer_validators:
            if not validator(trajectory, self.ground_truth_answers):
                return False
        return True
    
    def _compute_max_context_length(self, trajectory: MultiTurnTrajectory) -> int:
        """Compute maximum context length across all steps.
        
        Args:
            trajectory: Trajectory to analyze
            
        Returns:
            Maximum context length in tokens
        """
        if self.tokenizer is None:
            # Fallback to character count
            return max(len(step.context_input) for step in trajectory.steps)
        
        max_length = 0
        for step in trajectory.steps:
            tokens = self.tokenizer.encode(step.context_input, add_special_tokens=False)
            max_length = max(max_length, len(tokens))
        
        return max_length
    
    def _default_format_validator(self, trajectory: MultiTurnTrajectory) -> bool:
        """Default format validation logic.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            True if format is valid
        """
        if not trajectory.steps:
            return False
        
        # Check that all steps have non-empty responses
        for step in trajectory.steps:
            if not step.output_response.strip():
                return False
        
        # Check for basic structure indicators
        final_step = trajectory.steps[-1]
        if len(final_step.output_response) < 10:  # Too short to be meaningful
            return False
        
        return True
    
    def _default_answer_validator(self, 
                                 trajectory: MultiTurnTrajectory, 
                                 ground_truth_answers: Dict[str, Any]) -> bool:
        """Default answer validation logic.
        
        Args:
            trajectory: Trajectory to validate
            ground_truth_answers: Ground truth answers
            
        Returns:
            True if answer is correct
        """
        if not trajectory.steps:
            return False
        
        question_id = trajectory.question_id
        if question_id not in ground_truth_answers:
            # If no ground truth available, use heuristic validation
            return self._heuristic_answer_validation(trajectory)
        
        ground_truth = ground_truth_answers[question_id]
        final_response = trajectory.steps[-1].output_response
        
        # Extract answer from final response
        extracted_answer = self._extract_answer(final_response)
        
        # Compare with ground truth
        return self._compare_answers(extracted_answer, ground_truth)
    
    def _heuristic_answer_validation(self, trajectory: MultiTurnTrajectory) -> bool:
        """Heuristic validation when ground truth is not available.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            True if answer appears valid
        """
        final_response = trajectory.steps[-1].output_response.lower()
        
        # Check for answer indicators
        answer_indicators = [
            "the answer is",
            "final answer",
            "therefore",
            "conclusion",
            "result is"
        ]
        
        has_answer_indicator = any(indicator in final_response for indicator in answer_indicators)
        
        # Check for numeric answer (common in math problems)
        has_numeric_answer = bool(re.search(r'\d+', final_response))
        
        # Check minimum response length
        has_sufficient_length = len(final_response) >= 20
        
        return has_answer_indicator and (has_numeric_answer or has_sufficient_length)
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from a response.
        
        Args:
            response: Response text to extract answer from
            
        Returns:
            Extracted answer string
        """
        # Look for common answer patterns
        patterns = [
            r"the answer is\s*([^\n.]+)",
            r"final answer[:\s]*([^\n.]+)",
            r"therefore[,\s]*([^\n.]+)",
            r"answer[:\s]*([^\n.]+)"
        ]
        
        response_lower = response.lower()
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback: return last sentence
        sentences = response.split('.')
        return sentences[-1].strip() if sentences else response.strip()
    
    def _compare_answers(self, extracted: str, ground_truth: Any) -> bool:
        """Compare extracted answer with ground truth.
        
        Args:
            extracted: Extracted answer string
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match
        """
        if isinstance(ground_truth, str):
            return extracted.lower().strip() == ground_truth.lower().strip()
        elif isinstance(ground_truth, (int, float)):
            # Try to extract numeric value
            numeric_match = re.search(r'-?\d+\.?\d*', extracted)
            if numeric_match:
                try:
                    extracted_num = float(numeric_match.group())
                    return abs(extracted_num - ground_truth) < 1e-6
                except ValueError:
                    pass
        elif isinstance(ground_truth, list):
            # Multiple acceptable answers
            return any(self._compare_answers(extracted, gt) for gt in ground_truth)
        
        return False


class BatchTrajectoryRewardComputer:
    """Batch processor for computing trajectory rewards."""
    
    def __init__(self, reward_computer: TrajectoryRewardComputer):
        """Initialize batch processor.
        
        Args:
            reward_computer: Individual trajectory reward computer
        """
        self.reward_computer = reward_computer
    
    def compute_batch_rewards(self, 
                            trajectories: List[MultiTurnTrajectory]) -> torch.Tensor:
        """Compute rewards for a batch of trajectories.
        
        Args:
            trajectories: List of trajectories to evaluate
            
        Returns:
            Tensor of trajectory rewards
        """
        rewards = []
        for trajectory in trajectories:
            reward = self.reward_computer.compute_reward(trajectory)
            trajectory.trajectory_reward = reward
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def compute_grouped_rewards(self, 
                              trajectory_groups: List[List[MultiTurnTrajectory]]) -> Dict[str, torch.Tensor]:
        """Compute rewards for grouped trajectories (by question).
        
        Args:
            trajectory_groups: List of trajectory groups, one per question
            
        Returns:
            Dictionary containing reward tensors and metadata
        """
        all_trajectories = []
        group_sizes = []
        question_ids = []
        
        for group in trajectory_groups:
            all_trajectories.extend(group)
            group_sizes.append(len(group))
            if group:
                question_ids.append(group[0].question_id)
        
        # Compute all rewards
        all_rewards = self.compute_batch_rewards(all_trajectories)
        
        # Group rewards by question
        grouped_rewards = []
        start_idx = 0
        for group_size in group_sizes:
            group_rewards = all_rewards[start_idx:start_idx + group_size]
            grouped_rewards.append(group_rewards)
            start_idx += group_size
        
        return {
            'all_rewards': all_rewards,
            'grouped_rewards': grouped_rewards,
            'group_sizes': group_sizes,
            'question_ids': question_ids,
            'trajectories': all_trajectories
        }


def create_math_reward_computer(tokenizer: Optional[PreTrainedTokenizer] = None,
                               ground_truth_file: Optional[str] = None) -> GRPOTrajectoryRewardComputer:
    """Create a reward computer specialized for math problems.
    
    Args:
        tokenizer: Tokenizer for computing context lengths
        ground_truth_file: Path to file containing ground truth answers
        
    Returns:
        Configured reward computer for math problems
    """
    # Load ground truth answers if provided
    ground_truth_answers = {}
    if ground_truth_file:
        # TODO: Implement ground truth loading
        # This would load answers from a JSON/CSV file
        pass
    
    # Math-specific format validators
    def math_format_validator(trajectory: MultiTurnTrajectory) -> bool:
        """Validate format for math problems."""
        if not trajectory.steps:
            return False
        
        final_response = trajectory.steps[-1].output_response
        
        # Check for mathematical reasoning indicators
        math_indicators = ["calculate", "solve", "equation", "=", "+", "-", "*", "/"]
        has_math_content = any(indicator in final_response.lower() for indicator in math_indicators)
        
        # Check for step-by-step reasoning
        has_steps = len(final_response.split('\n')) > 1 or 'step' in final_response.lower()
        
        return has_math_content and has_steps
    
    # Math-specific answer validators
    def math_answer_validator(trajectory: MultiTurnTrajectory, 
                            ground_truth_answers: Dict[str, Any]) -> bool:
        """Validate answers for math problems."""
        if not trajectory.steps:
            return False
        
        final_response = trajectory.steps[-1].output_response
        
        # Look for final numeric answer
        numeric_pattern = r'(?:answer is|equals?|=)\s*(-?\d+(?:\.\d+)?)'
        match = re.search(numeric_pattern, final_response.lower())
        
        if not match:
            # Fallback: look for any number at the end
            end_number = re.search(r'(-?\d+(?:\.\d+)?)\s*$', final_response.strip())
            if not end_number:
                return False
        
        return True
    
    return GRPOTrajectoryRewardComputer(
        tokenizer=tokenizer,
        ground_truth_answers=ground_truth_answers,
        format_validators=[math_format_validator],
        answer_validators=[math_answer_validator]
    )


def create_coding_reward_computer(tokenizer: Optional[PreTrainedTokenizer] = None,
                                ground_truth_file: Optional[str] = None) -> GRPOTrajectoryRewardComputer:
    """Create a reward computer specialized for coding problems.
    
    Args:
        tokenizer: Tokenizer for computing context lengths
        ground_truth_file: Path to file containing ground truth answers
        
    Returns:
        Configured reward computer for coding problems
    """
    # Load ground truth answers if provided
    ground_truth_answers = {}
    if ground_truth_file:
        # TODO: Implement ground truth loading
        pass
    
    # Coding-specific format validators
    def coding_format_validator(trajectory: MultiTurnTrajectory) -> bool:
        """Validate format for coding problems."""
        if not trajectory.steps:
            return False
        
        final_response = trajectory.steps[-1].output_response
        
        # Check for code blocks
        has_code_block = '```' in final_response or 'def ' in final_response or 'class ' in final_response
        
        # Check for programming keywords
        programming_keywords = ['function', 'variable', 'loop', 'if', 'else', 'return', 'import']
        has_programming_content = any(keyword in final_response.lower() for keyword in programming_keywords)
        
        return has_code_block or has_programming_content
    
    # Coding-specific answer validators
    def coding_answer_validator(trajectory: MultiTurnTrajectory, 
                              ground_truth_answers: Dict[str, Any]) -> bool:
        """Validate answers for coding problems."""
        if not trajectory.steps:
            return False
        
        final_response = trajectory.steps[-1].output_response
        
        # Check for executable code
        has_executable_code = ('def ' in final_response or 
                             'class ' in final_response or 
                             'return ' in final_response)
        
        # Check for proper code structure
        has_proper_structure = final_response.count('(') == final_response.count(')')
        
        return has_executable_code and has_proper_structure
    
    return GRPOTrajectoryRewardComputer(
        tokenizer=tokenizer,
        ground_truth_answers=ground_truth_answers,
        format_validators=[coding_format_validator],
        answer_validators=[coding_answer_validator]
    )