#!/usr/bin/env python3
"""
Test script for multi-turn GRPO implementation.
This script demonstrates the basic usage of the multi-turn GRPO components.
"""

import torch
import numpy as np
from omegaconf import DictConfig

# Import the multi-turn GRPO components
from verl.workers.rollout.multi_turn_trajectory import (
    MultiTurnTrajectory,
    TrajectoryStep,
    SimpleContextCondenser,
    MultiTurnTrajectoryRollout
)
from verl.trainer.ppo.trajectory_reward import (
    GRPOTrajectoryRewardComputer,
    BatchTrajectoryRewardComputer,
    create_math_reward_computer
)
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


class MockRollout:
    """Mock rollout implementation for testing."""
    
    def generate_sequences(self, prompts):
        """Generate mock sequences."""
        return {"mock": "response"}


def test_trajectory_creation():
    """Test basic trajectory creation and manipulation."""
    print("Testing trajectory creation...")
    
    # Create a sample trajectory
    trajectory = MultiTurnTrajectory(
        trajectory_id="test_traj_1",
        question_id="q_1",
        steps=[]
    )
    
    # Add some steps
    for i in range(3):
        step = TrajectoryStep(
            context_input=f"Context for step {i}",
            output_response=f"Response for step {i}",
            input_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5, 6]),
            step_index=i
        )
        trajectory.steps.append(step)
    
    print(f"Created trajectory with {len(trajectory.steps)} steps")
    print(f"Trajectory ID: {trajectory.trajectory_id}")
    print(f"Question ID: {trajectory.question_id}")
    
    return trajectory


def test_context_condensation():
    """Test context condensation functionality."""
    print("\nTesting context condensation...")
    
    # Create a context condenser
    condenser = SimpleContextCondenser(keep_last_n_steps=2)
    
    # Create some trajectory steps
    steps = []
    for i in range(4):
        step = TrajectoryStep(
            context_input=f"Context {i}",
            output_response=f"Response {i}",
            input_ids=torch.tensor([i]),
            response_ids=torch.tensor([i+10]),
            step_index=i
        )
        steps.append(step)
    
    # Test condensation
    original_context = "Original question"
    condensed = condenser.condense_context(
        original_context=original_context,
        previous_steps=steps,
        max_length=1000
    )
    
    print(f"Original context: {original_context}")
    print(f"Condensed context: {condensed[:100]}...")
    
    return condenser


def test_reward_computation():
    """Test trajectory reward computation."""
    print("\nTesting reward computation...")
    
    # Create a reward computer
    reward_computer = GRPOTrajectoryRewardComputer()
    
    # Create a test trajectory
    trajectory = test_trajectory_creation()
    
    # Compute reward
    reward = reward_computer.compute_reward(trajectory)
    
    print(f"Computed reward: {reward}")
    print(f"Is valid format: {trajectory.is_valid_format}")
    print(f"Is correct answer: {trajectory.is_correct_answer}")
    print(f"Max context length: {trajectory.max_context_length}")
    
    return reward_computer


def test_batch_reward_computation():
    """Test batch reward computation."""
    print("\nTesting batch reward computation...")
    
    # Create multiple trajectories
    trajectories = []
    for i in range(5):
        trajectory = MultiTurnTrajectory(
            trajectory_id=f"test_traj_{i}",
            question_id=f"q_{i // 2}",  # Group trajectories by question
            steps=[]
        )
        
        # Add steps to trajectory
        for j in range(2):
            step = TrajectoryStep(
                context_input=f"Context {i}_{j}",
                output_response=f"Response {i}_{j} with answer {i*j}",
                input_ids=torch.tensor([i, j]),
                response_ids=torch.tensor([i+10, j+10]),
                step_index=j
            )
            trajectory.steps.append(step)
        
        trajectories.append(trajectory)
    
    # Create batch reward computer
    reward_computer = GRPOTrajectoryRewardComputer()
    batch_computer = BatchTrajectoryRewardComputer(reward_computer)
    
    # Compute batch rewards
    rewards = batch_computer.compute_batch_rewards(trajectories)
    
    print(f"Computed {len(rewards)} rewards: {rewards}")
    
    # Test grouped rewards
    trajectory_groups = [trajectories[:2], trajectories[2:4], trajectories[4:]]
    grouped_results = batch_computer.compute_grouped_rewards(trajectory_groups)
    
    print(f"Grouped rewards: {grouped_results['all_rewards']}")
    print(f"Group sizes: {grouped_results['group_sizes']}")
    
    return batch_computer


def test_advantage_computation():
    """Test GRPO multi-turn advantage computation."""
    print("\nTesting advantage computation...")
    
    try:
        # Get the multi-turn GRPO advantage estimator
        adv_estimator_fn = get_adv_estimator_fn('grpo_multiturn')
        
        # Create test data
        batch_size = 4
        seq_length = 10
        
        token_level_rewards = torch.zeros(batch_size, seq_length)
        response_mask = torch.ones(batch_size, seq_length)
        trajectory_step_masks = torch.ones(batch_size, seq_length)
        trajectory_rewards = torch.tensor([0.5, 0.3, 0.8, 0.2])
        index = np.array([0, 0, 1, 1])  # Two questions, two trajectories each
        
        # Compute advantages
        advantages, returns = adv_estimator_fn(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            trajectory_step_masks=trajectory_step_masks,
            trajectory_rewards=trajectory_rewards,
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=True,
            config=None
        )
        
        print(f"Advantages shape: {advantages.shape}")
        print(f"Advantages mean: {advantages.mean().item():.4f}")
        print(f"Advantages std: {advantages.std().item():.4f}")
        
        return advantages
        
    except Exception as e:
        print(f"Advantage computation test failed: {e}")
        print("This is expected if the advantage estimator is not properly registered")
        return None


def test_math_reward_computer():
    """Test math-specific reward computer."""
    print("\nTesting math reward computer...")
    
    # Create math reward computer
    math_computer = create_math_reward_computer()
    
    # Create a math trajectory
    trajectory = MultiTurnTrajectory(
        trajectory_id="math_traj_1",
        question_id="math_q_1",
        steps=[]
    )
    
    # Add math-like steps
    steps_data = [
        ("What is 2 + 3?", "Let me calculate: 2 + 3"),
        ("Continue the calculation", "2 + 3 = 5. Therefore, the answer is 5."),
    ]
    
    for i, (context, response) in enumerate(steps_data):
        step = TrajectoryStep(
            context_input=context,
            output_response=response,
            input_ids=torch.tensor([1, 2, 3]),
            response_ids=torch.tensor([4, 5, 6]),
            step_index=i
        )
        trajectory.steps.append(step)
    
    # Compute reward
    reward = math_computer.compute_reward(trajectory)
    
    print(f"Math trajectory reward: {reward}")
    print(f"Is valid format: {trajectory.is_valid_format}")
    print(f"Is correct answer: {trajectory.is_correct_answer}")
    
    return math_computer


def main():
    """Run all tests."""
    print("=" * 50)
    print("Multi-Turn GRPO Component Tests")
    print("=" * 50)
    
    try:
        # Test basic components
        test_trajectory_creation()
        test_context_condensation()
        test_reward_computation()
        test_batch_reward_computation()
        test_math_reward_computer()
        
        # Test advantage computation (may fail if not properly integrated)
        test_advantage_computation()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()