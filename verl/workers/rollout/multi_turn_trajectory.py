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
Multi-turn trajectory rollout logic for GRPO with context condensation.
This module implements the rollout logic for generating multiple trajectories
per question with context condensation at each turn.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import AsyncRolloutRequest


@dataclass
class TrajectoryStep:
    """Represents a single step in a multi-turn trajectory.
    
    Attributes:
        context_input (str): The input context/prompt for this step (ci)
        output_response (str): The model's output response for this step (oi)
        input_ids (torch.Tensor): Tokenized input IDs
        response_ids (torch.Tensor): Tokenized response IDs
        step_index (int): Index of this step in the trajectory
        condensed_context (Optional[str]): Context after condensation for next step
    """
    context_input: str
    output_response: str
    input_ids: torch.Tensor
    response_ids: torch.Tensor
    step_index: int
    condensed_context: Optional[str] = None


@dataclass
class MultiTurnTrajectory:
    """Represents a complete multi-turn trajectory.
    
    Attributes:
        trajectory_id (str): Unique identifier for this trajectory
        question_id (str): ID of the original question
        steps (List[TrajectoryStep]): List of trajectory steps [(c1,o1), (c2,o2), ...]
        trajectory_reward (float): Final reward for the complete trajectory
        max_context_length (int): Maximum context length used in this trajectory
        is_valid_format (bool): Whether the trajectory has valid format
        is_correct_answer (bool): Whether the final answer is correct
    """
    trajectory_id: str
    question_id: str
    steps: List[TrajectoryStep]
    trajectory_reward: float = 0.0
    max_context_length: int = 0
    is_valid_format: bool = True
    is_correct_answer: bool = False


class ContextCondenser(ABC):
    """Abstract base class for context condensation strategies."""
    
    @abstractmethod
    def condense_context(self, 
                        original_context: str, 
                        previous_steps: List[TrajectoryStep],
                        max_length: int) -> str:
        """Condense context for the next turn.
        
        Args:
            original_context: The original question/context
            previous_steps: Previous steps in the trajectory
            max_length: Maximum allowed context length
            
        Returns:
            Condensed context string for the next turn
        """
        pass


class SimpleContextCondenser(ContextCondenser):
    """Simple context condensation that keeps only the most recent interactions."""
    
    def __init__(self, keep_last_n_steps: int = 2):
        """Initialize the condenser.
        
        Args:
            keep_last_n_steps: Number of recent steps to keep in context
        """
        self.keep_last_n_steps = keep_last_n_steps
    
    def condense_context(self, 
                        original_context: str, 
                        previous_steps: List[TrajectoryStep],
                        max_length: int) -> str:
        """Condense context by keeping only recent steps.
        
        Args:
            original_context: The original question/context
            previous_steps: Previous steps in the trajectory
            max_length: Maximum allowed context length
            
        Returns:
            Condensed context string
        """
        if not previous_steps:
            return original_context
        
        # Keep the original question and last N steps
        recent_steps = previous_steps[-self.keep_last_n_steps:]
        
        condensed = original_context
        for step in recent_steps:
            condensed += f"\n\nPrevious attempt: {step.context_input}\nResponse: {step.output_response}"
        
        # TODO: Implement proper length truncation based on tokenizer
        # For now, simple character-based truncation
        if len(condensed) > max_length:
            condensed = condensed[:max_length]
        
        return condensed


class MultiTurnTrajectoryRollout(BaseRollout):
    """Multi-turn trajectory rollout implementation for GRPO.
    
    This class generates multiple trajectories per question, where each trajectory
    consists of multiple turns with context condensation between turns.
    """
    
    def __init__(self,
                 base_rollout: BaseRollout,
                 context_condenser: Optional[ContextCondenser] = None,
                 max_turns_per_trajectory: int = 5,
                 n_trajectories_per_question: int = 4,
                 max_context_length: int = 2048):
        """Initialize the multi-turn trajectory rollout.
        
        Args:
            base_rollout: Base rollout implementation to use for generation
            context_condenser: Strategy for condensing context between turns
            max_turns_per_trajectory: Maximum number of turns per trajectory
            n_trajectories_per_question: Number of trajectories to generate per question (GRPO group size)
            max_context_length: Maximum context length for condensation
        """
        self.base_rollout = base_rollout
        self.context_condenser = context_condenser or SimpleContextCondenser()
        self.max_turns_per_trajectory = max_turns_per_trajectory
        self.n_trajectories_per_question = n_trajectories_per_question
        self.max_context_length = max_context_length
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate multi-turn trajectory sequences.
        
        Args:
            prompts: Input prompts/questions
            
        Returns:
            DataProto containing generated trajectories
        """
        # TODO: Implement the main generation logic
        # This is a skeleton implementation
        trajectories = []
        
        for prompt_idx, prompt in enumerate(prompts):
            question_id = f"q_{prompt_idx}"
            
            # Generate n trajectories for this question
            for traj_idx in range(self.n_trajectories_per_question):
                trajectory_id = f"{question_id}_traj_{traj_idx}"
                trajectory = self._generate_single_trajectory(
                    question_id=question_id,
                    trajectory_id=trajectory_id,
                    initial_prompt=prompt
                )
                trajectories.append(trajectory)
        
        return self._convert_trajectories_to_dataproto(trajectories)
    
    def _generate_single_trajectory(self, 
                                   question_id: str,
                                   trajectory_id: str, 
                                   initial_prompt: str) -> MultiTurnTrajectory:
        """Generate a single multi-turn trajectory.
        
        Args:
            question_id: ID of the original question
            trajectory_id: Unique ID for this trajectory
            initial_prompt: Initial prompt/question
            
        Returns:
            Complete multi-turn trajectory
        """
        trajectory = MultiTurnTrajectory(
            trajectory_id=trajectory_id,
            question_id=question_id,
            steps=[]
        )
        
        current_context = initial_prompt
        
        for turn_idx in range(self.max_turns_per_trajectory):
            # Generate response for current context
            step = self._generate_trajectory_step(
                context_input=current_context,
                step_index=turn_idx,
                trajectory_id=trajectory_id
            )
            
            trajectory.steps.append(step)
            
            # Check if trajectory should terminate (e.g., final answer reached)
            if self._should_terminate_trajectory(step, trajectory):
                break
            
            # Condense context for next turn
            current_context = self.context_condenser.condense_context(
                original_context=initial_prompt,
                previous_steps=trajectory.steps,
                max_length=self.max_context_length
            )
        
        # Compute trajectory-level reward
        trajectory.trajectory_reward = self._compute_trajectory_reward(trajectory)
        
        return trajectory
    
    def _generate_trajectory_step(self, 
                                 context_input: str, 
                                 step_index: int,
                                 trajectory_id: str) -> TrajectoryStep:
        """Generate a single step in the trajectory.
        
        Args:
            context_input: Input context for this step
            step_index: Index of this step in the trajectory
            trajectory_id: ID of the parent trajectory
            
        Returns:
            Generated trajectory step
        """
        # TODO: Implement actual generation using base_rollout
        # This is a skeleton implementation
        
        # For now, create a dummy response
        output_response = f"[Step {step_index}] Generated response for: {context_input[:50]}..."
        
        # TODO: Tokenize inputs and outputs properly
        input_ids = torch.tensor([1, 2, 3])  # Dummy tokenization
        response_ids = torch.tensor([4, 5, 6])  # Dummy tokenization
        
        return TrajectoryStep(
            context_input=context_input,
            output_response=output_response,
            input_ids=input_ids,
            response_ids=response_ids,
            step_index=step_index
        )
    
    def _should_terminate_trajectory(self, 
                                   step: TrajectoryStep, 
                                   trajectory: MultiTurnTrajectory) -> bool:
        """Determine if trajectory should terminate after this step.
        
        Args:
            step: Current trajectory step
            trajectory: Current trajectory state
            
        Returns:
            True if trajectory should terminate
        """
        # TODO: Implement termination logic
        # For now, simple heuristic based on response content
        
        # Terminate if response contains final answer indicators
        final_answer_indicators = ["final answer", "the answer is", "therefore"]
        return any(indicator in step.output_response.lower() 
                  for indicator in final_answer_indicators)
    
    def _compute_trajectory_reward(self, trajectory: MultiTurnTrajectory) -> float:
        """Compute reward for a complete trajectory.
        
        Reward logic:
        - 0 for bad format
        - 0 for wrong answer  
        - reward = 1 / max_context_length otherwise
        
        Args:
            trajectory: Complete trajectory to evaluate
            
        Returns:
            Trajectory reward value
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
        max_context_length = max(len(step.context_input) for step in trajectory.steps)
        trajectory.max_context_length = max_context_length
        trajectory.is_correct_answer = True
        
        return 1.0 / max(max_context_length, 1)  # Avoid division by zero
    
    def _is_valid_format(self, trajectory: MultiTurnTrajectory) -> bool:
        """Check if trajectory has valid format.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            True if format is valid
        """
        # TODO: Implement format validation logic
        # For now, simple checks
        
        if not trajectory.steps:
            return False
        
        # Check that all steps have non-empty responses
        return all(step.output_response.strip() for step in trajectory.steps)
    
    def _is_correct_answer(self, trajectory: MultiTurnTrajectory) -> bool:
        """Check if trajectory produces correct answer.
        
        Args:
            trajectory: Trajectory to evaluate
            
        Returns:
            True if answer is correct
        """
        # TODO: Implement answer correctness checking
        # This would typically involve comparing with ground truth
        # For now, placeholder implementation
        
        if not trajectory.steps:
            return False
        
        # Simple heuristic: check if final response contains numeric answer
        final_response = trajectory.steps[-1].output_response
        return any(char.isdigit() for char in final_response)
    
    def _convert_trajectories_to_dataproto(self, 
                                         trajectories: List[MultiTurnTrajectory]) -> DataProto:
        """Convert trajectories to DataProto format.
        
        Args:
            trajectories: List of generated trajectories
            
        Returns:
            DataProto containing trajectory data
        """
        # TODO: Implement proper conversion to DataProto
        # This is a skeleton implementation
        
        # For now, return a simple structure
        return {
            'trajectories': trajectories,
            'n_trajectories': len(trajectories),
            'trajectory_groups': self._group_trajectories_by_question(trajectories)
        }
    
    def _group_trajectories_by_question(self, 
                                      trajectories: List[MultiTurnTrajectory]) -> Dict[str, List[MultiTurnTrajectory]]:
        """Group trajectories by their original question ID.
        
        Args:
            trajectories: List of trajectories to group
            
        Returns:
            Dictionary mapping question IDs to trajectory lists
        """
        groups = {}
        for trajectory in trajectories:
            question_id = trajectory.question_id
            if question_id not in groups:
                groups[question_id] = []
            groups[question_id].append(trajectory)
        
        return groups


class MultiTurnTrajectoryManager:
    """Manager class for handling multi-turn trajectory operations."""
    
    def __init__(self, rollout: MultiTurnTrajectoryRollout):
        """Initialize the trajectory manager.
        
        Args:
            rollout: Multi-turn trajectory rollout instance
        """
        self.rollout = rollout
    
    def generate_trajectory_batch(self, 
                                questions: List[str],
                                batch_size: int = 32) -> List[List[MultiTurnTrajectory]]:
        """Generate trajectories for a batch of questions.
        
        Args:
            questions: List of input questions
            batch_size: Batch size for processing
            
        Returns:
            List of trajectory groups, one per question
        """
        # TODO: Implement batched trajectory generation
        trajectory_groups = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_results = self.rollout.generate_sequences(batch_questions)
            
            # Extract trajectory groups from results
            if 'trajectory_groups' in batch_results:
                trajectory_groups.extend(batch_results['trajectory_groups'].values())
        
        return trajectory_groups
    
    def compute_grpo_advantages(self, 
                              trajectory_groups: List[List[MultiTurnTrajectory]]) -> torch.Tensor:
        """Compute GRPO advantages for trajectory groups.
        
        Args:
            trajectory_groups: Groups of trajectories per question
            
        Returns:
            Advantage tensor for all trajectories
        """
        # TODO: This will be implemented in the next task
        # For now, return placeholder
        total_trajectories = sum(len(group) for group in trajectory_groups)
        return torch.zeros(total_trajectories)