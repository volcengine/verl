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
Multi-turn GRPO rollout worker integration.
This module integrates multi-turn trajectory generation with existing VERL workers.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.trainer.ppo.trajectory_reward import (
    BatchTrajectoryRewardComputer,
    GRPOTrajectoryRewardComputer,
    create_math_reward_computer,
    create_coding_reward_computer
)
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.multi_turn_trajectory import (
    MultiTurnTrajectory,
    MultiTurnTrajectoryRollout,
    MultiTurnTrajectoryManager,
    SimpleContextCondenser
)


class MultiTurnGRPORolloutWorker:
    """
    Worker class that integrates multi-turn trajectory generation with GRPO training.
    
    This worker handles:
    1. Generating n trajectories per question
    2. Computing trajectory-level rewards
    3. Preparing data for GRPO advantage computation
    4. Integration with existing VERL training pipeline
    """
    
    def __init__(self,
                 base_rollout: BaseRollout,
                 config: DictConfig,
                 tokenizer: Optional[Any] = None):
        """Initialize the multi-turn GRPO rollout worker.
        
        Args:
            base_rollout: Base rollout implementation (e.g., VLLM, SGLang)
            config: Configuration for multi-turn GRPO
            tokenizer: Tokenizer for text processing
        """
        self.base_rollout = base_rollout
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize multi-turn trajectory components
        self._init_trajectory_components()
        
        # Initialize reward computation
        self._init_reward_computation()
    
    def _init_trajectory_components(self):
        """Initialize trajectory generation components."""
        # Context condensation strategy
        condenser_config = self.config.get('context_condenser', {})
        condenser_type = condenser_config.get('type', 'simple')
        
        if condenser_type == 'simple':
            keep_last_n = condenser_config.get('keep_last_n_steps', 2)
            self.context_condenser = SimpleContextCondenser(keep_last_n_steps=keep_last_n)
        else:
            # TODO: Add support for other condensation strategies
            self.context_condenser = SimpleContextCondenser()
        
        # Multi-turn trajectory rollout
        self.trajectory_rollout = MultiTurnTrajectoryRollout(
            base_rollout=self.base_rollout,
            context_condenser=self.context_condenser,
            max_turns_per_trajectory=self.config.get('max_turns_per_trajectory', 5),
            n_trajectories_per_question=self.config.get('n_trajectories_per_question', 4),
            max_context_length=self.config.get('max_context_length', 2048)
        )
        
        # Trajectory manager
        self.trajectory_manager = MultiTurnTrajectoryManager(self.trajectory_rollout)
    
    def _init_reward_computation(self):
        """Initialize reward computation components."""
        reward_config = self.config.get('reward_computation', {})
        reward_type = reward_config.get('type', 'grpo')
        domain = reward_config.get('domain', 'general')
        
        if reward_type == 'grpo':
            if domain == 'math':
                self.reward_computer = create_math_reward_computer(
                    tokenizer=self.tokenizer,
                    ground_truth_file=reward_config.get('ground_truth_file')
                )
            elif domain == 'coding':
                self.reward_computer = create_coding_reward_computer(
                    tokenizer=self.tokenizer,
                    ground_truth_file=reward_config.get('ground_truth_file')
                )
            else:
                self.reward_computer = GRPOTrajectoryRewardComputer(
                    tokenizer=self.tokenizer,
                    ground_truth_answers=reward_config.get('ground_truth_answers', {})
                )
        else:
            # TODO: Add support for other reward computation strategies
            self.reward_computer = GRPOTrajectoryRewardComputer(tokenizer=self.tokenizer)
        
        # Batch reward computer
        self.batch_reward_computer = BatchTrajectoryRewardComputer(self.reward_computer)
    
    def generate_trajectories(self, prompts: DataProto) -> Dict[str, Any]:
        """Generate multi-turn trajectories for given prompts.
        
        Args:
            prompts: Input prompts/questions
            
        Returns:
            Dictionary containing generated trajectories and metadata
        """
        # Extract questions from prompts
        questions = self._extract_questions_from_prompts(prompts)
        
        # Generate trajectory groups (n trajectories per question)
        trajectory_groups = self.trajectory_manager.generate_trajectory_batch(
            questions=questions,
            batch_size=self.config.get('batch_size', 32)
        )
        
        # Compute trajectory rewards
        reward_results = self.batch_reward_computer.compute_grouped_rewards(trajectory_groups)
        
        # Prepare data for GRPO training
        training_data = self._prepare_training_data(
            trajectory_groups=trajectory_groups,
            reward_results=reward_results
        )
        
        return {
            'trajectory_groups': trajectory_groups,
            'reward_results': reward_results,
            'training_data': training_data,
            'metadata': {
                'n_questions': len(questions),
                'n_trajectories_per_question': self.config.get('n_trajectories_per_question', 4),
                'total_trajectories': len(reward_results['trajectories'])
            }
        }
    
    def _extract_questions_from_prompts(self, prompts: DataProto) -> List[str]:
        """Extract question strings from DataProto prompts.
        
        Args:
            prompts: Input prompts in DataProto format
            
        Returns:
            List of question strings
        """
        # TODO: Implement proper extraction based on DataProto structure
        # This is a skeleton implementation
        
        if isinstance(prompts, dict) and 'questions' in prompts:
            return prompts['questions']
        elif isinstance(prompts, list):
            return [str(prompt) for prompt in prompts]
        else:
            # Fallback: assume single prompt
            return [str(prompts)]
    
    def _prepare_training_data(self, 
                             trajectory_groups: List[List[MultiTurnTrajectory]],
                             reward_results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare training data for GRPO multi-turn advantage computation.
        
        Args:
            trajectory_groups: Groups of trajectories per question
            reward_results: Results from reward computation
            
        Returns:
            Dictionary containing tensors for GRPO training
        """
        trajectories = reward_results['trajectories']
        all_rewards = reward_results['all_rewards']
        
        # Prepare tensors for advantage computation
        batch_size = len(trajectories)
        max_length = self.config.get('max_sequence_length', 2048)
        
        # Initialize tensors
        token_level_rewards = torch.zeros(batch_size, max_length)
        response_mask = torch.zeros(batch_size, max_length)
        trajectory_step_masks = torch.zeros(batch_size, max_length)
        
        # Create grouping index for GRPO
        question_ids = [traj.question_id for traj in trajectories]
        unique_questions = list(set(question_ids))
        question_to_idx = {q: i for i, q in enumerate(unique_questions)}
        index = np.array([question_to_idx[q] for q in question_ids])
        
        # Process each trajectory
        for i, trajectory in enumerate(trajectories):
            # TODO: Implement proper tokenization and mask creation
            # This is a skeleton implementation
            
            # Tokenize trajectory steps
            input_ids, attention_mask, step_mask = self._tokenize_trajectory(trajectory, max_length)
            
            # Set masks
            response_mask[i, :len(attention_mask)] = torch.tensor(attention_mask)
            trajectory_step_masks[i, :len(step_mask)] = torch.tensor(step_mask)
        
        return {
            'token_level_rewards': token_level_rewards,
            'response_mask': response_mask,
            'trajectory_step_masks': trajectory_step_masks,
            'trajectory_rewards': all_rewards,
            'index': index,
            'trajectories': trajectories
        }
    
    def _tokenize_trajectory(self, 
                           trajectory: MultiTurnTrajectory, 
                           max_length: int) -> tuple[List[int], List[int], List[int]]:
        """Tokenize a trajectory and create masks.
        
        Args:
            trajectory: Trajectory to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (input_ids, attention_mask, step_mask)
            step_mask: 1 for output tokens (oi), 0 for input context tokens (ci)
        """
        # TODO: Implement proper tokenization
        # This is a skeleton implementation
        
        if self.tokenizer is None:
            # Fallback: create dummy tokens
            dummy_length = min(100, max_length)
            input_ids = list(range(dummy_length))
            attention_mask = [1] * dummy_length
            step_mask = [1] * dummy_length  # All tokens treated as output for now
            return input_ids, attention_mask, step_mask
        
        # Concatenate all trajectory steps
        full_text = ""
        step_boundaries = []
        
        for step in trajectory.steps:
            context_start = len(full_text)
            full_text += step.context_input
            context_end = len(full_text)
            
            response_start = len(full_text)
            full_text += " " + step.output_response
            response_end = len(full_text)
            
            step_boundaries.append({
                'context_range': (context_start, context_end),
                'response_range': (response_start, response_end)
            })
        
        # Tokenize full text
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True, max_length=max_length, truncation=True)
        attention_mask = [1] * len(tokens)
        
        # Create step mask (1 for output tokens, 0 for input context)
        # TODO: Implement proper character-to-token mapping
        step_mask = [1] * len(tokens)  # Simplified: all tokens as output
        
        return tokens, attention_mask, step_mask
    
    def compute_advantages(self, training_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute GRPO advantages for multi-turn trajectories.
        
        Args:
            training_data: Prepared training data
            
        Returns:
            Advantage tensor
        """
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn
        
        # Get the multi-turn GRPO advantage estimator
        adv_estimator_fn = get_adv_estimator_fn('grpo_multiturn')
        
        # Compute advantages
        advantages, returns = adv_estimator_fn(
            token_level_rewards=training_data['token_level_rewards'],
            response_mask=training_data['response_mask'],
            index=training_data['index'],
            trajectory_step_masks=training_data['trajectory_step_masks'],
            trajectory_rewards=training_data['trajectory_rewards'],
            epsilon=self.config.get('advantage_epsilon', 1e-6),
            norm_adv_by_std_in_grpo=self.config.get('norm_adv_by_std', True),
            config=self.config
        )
        
        return advantages
    
    def rollout_and_compute_advantages(self, prompts: DataProto) -> Dict[str, Any]:
        """Complete pipeline: generate trajectories and compute advantages.
        
        Args:
            prompts: Input prompts/questions
            
        Returns:
            Dictionary containing trajectories, rewards, and advantages
        """
        # Generate trajectories
        trajectory_results = self.generate_trajectories(prompts)
        
        # Compute advantages
        advantages = self.compute_advantages(trajectory_results['training_data'])
        
        return {
            **trajectory_results,
            'advantages': advantages,
            'pipeline_metadata': {
                'advantage_shape': advantages.shape,
                'advantage_mean': advantages.mean().item(),
                'advantage_std': advantages.std().item()
            }
        }


class MultiTurnGRPOWorkerFactory:
    """Factory for creating multi-turn GRPO workers with different configurations."""
    
    @staticmethod
    def create_math_worker(base_rollout: BaseRollout, 
                          tokenizer: Optional[Any] = None,
                          n_trajectories: int = 4,
                          max_turns: int = 5) -> MultiTurnGRPORolloutWorker:
        """Create a worker specialized for math problems.
        
        Args:
            base_rollout: Base rollout implementation
            tokenizer: Tokenizer for text processing
            n_trajectories: Number of trajectories per question
            max_turns: Maximum turns per trajectory
            
        Returns:
            Configured multi-turn GRPO worker for math problems
        """
        config = DictConfig({
            'n_trajectories_per_question': n_trajectories,
            'max_turns_per_trajectory': max_turns,
            'max_context_length': 2048,
            'context_condenser': {
                'type': 'simple',
                'keep_last_n_steps': 2
            },
            'reward_computation': {
                'type': 'grpo',
                'domain': 'math'
            },
            'batch_size': 32,
            'max_sequence_length': 2048,
            'advantage_epsilon': 1e-6,
            'norm_adv_by_std': True
        })
        
        return MultiTurnGRPORolloutWorker(
            base_rollout=base_rollout,
            config=config,
            tokenizer=tokenizer
        )
    
    @staticmethod
    def create_coding_worker(base_rollout: BaseRollout,
                           tokenizer: Optional[Any] = None,
                           n_trajectories: int = 4,
                           max_turns: int = 3) -> MultiTurnGRPORolloutWorker:
        """Create a worker specialized for coding problems.
        
        Args:
            base_rollout: Base rollout implementation
            tokenizer: Tokenizer for text processing
            n_trajectories: Number of trajectories per question
            max_turns: Maximum turns per trajectory
            
        Returns:
            Configured multi-turn GRPO worker for coding problems
        """
        config = DictConfig({
            'n_trajectories_per_question': n_trajectories,
            'max_turns_per_trajectory': max_turns,
            'max_context_length': 4096,  # Longer context for code
            'context_condenser': {
                'type': 'simple',
                'keep_last_n_steps': 1  # Keep less history for code
            },
            'reward_computation': {
                'type': 'grpo',
                'domain': 'coding'
            },
            'batch_size': 16,  # Smaller batch for longer sequences
            'max_sequence_length': 4096,
            'advantage_epsilon': 1e-6,
            'norm_adv_by_std': True
        })
        
        return MultiTurnGRPORolloutWorker(
            base_rollout=base_rollout,
            config=config,
            tokenizer=tokenizer
        )
    
    @staticmethod
    def create_general_worker(base_rollout: BaseRollout,
                            config: DictConfig,
                            tokenizer: Optional[Any] = None) -> MultiTurnGRPORolloutWorker:
        """Create a general-purpose multi-turn GRPO worker.
        
        Args:
            base_rollout: Base rollout implementation
            config: Custom configuration
            tokenizer: Tokenizer for text processing
            
        Returns:
            Configured multi-turn GRPO worker
        """
        return MultiTurnGRPORolloutWorker(
            base_rollout=base_rollout,
            config=config,
            tokenizer=tokenizer
        )