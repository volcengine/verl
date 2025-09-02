# Multi-Turn GRPO Trainer

This directory contains the implementation and examples for Multi-Turn GRPO (Group Relative Policy Optimization) with context condensation. This approach extends the standard GRPO algorithm to handle multi-turn trajectories where each turn has different input prompts due to context condensation.

## Overview

### Key Features

1. **Multi-Turn Trajectory Generation**: For each input question, generate `n` trajectories where each trajectory consists of multiple turns `[(c1,o1), (c2,o2), ..., (ck,ok)]`
2. **Context Condensation**: Instead of direct concatenation of history, each turn uses condensed context to manage memory and improve efficiency
3. **Trajectory-Level Rewards**: Compute a single reward for the entire trajectory based on format validity, answer correctness, and context efficiency
4. **GRPO-Style Advantage Estimation**: Apply the same advantage to each step in a trajectory while masking input context tokens (ci)

### Architecture

```
Question x → [Trajectory 1: [(c1,o1), (c2,o2), ...], 
              Trajectory 2: [(c1,o1), (c2,o2), ...], 
              ..., 
              Trajectory n: [(c1,o1), (c2,o2), ...]]
              
Each trajectory → Single reward R = {
    0 if bad format,
    0 if wrong answer,
    1/max_context_length otherwise
}

GRPO Advantage → Applied to all output tokens (oi) in trajectory, masked for input tokens (ci)
```

## Implementation Components

### 1. Multi-Turn Trajectory Generation (`multi_turn_trajectory.py`)

- **`MultiTurnTrajectory`**: Data structure representing a complete trajectory
- **`TrajectoryStep`**: Individual step in a trajectory with context input and output response
- **`ContextCondenser`**: Abstract base class for context condensation strategies
- **`MultiTurnTrajectoryRollout`**: Main rollout implementation for generating trajectories

### 2. GRPO Advantage Estimation (`core_algos.py`)

- **`compute_grpo_multiturn_advantage`**: Extended GRPO advantage computation for multi-turn trajectories
- **`prepare_multiturn_trajectory_data`**: Data preparation utilities for GRPO training

### 3. Trajectory Reward Computation (`trajectory_reward.py`)

- **`GRPOTrajectoryRewardComputer`**: Implements the reward logic (0 for bad format/wrong answer, 1/max_context_length otherwise)
- **`BatchTrajectoryRewardComputer`**: Batch processing for trajectory rewards
- **Domain-specific reward computers**: Specialized for math and coding problems

### 4. Worker Integration (`multi_turn_grpo_worker.py`)

- **`MultiTurnGRPORolloutWorker`**: Integrates trajectory generation with existing VERL workers
- **`MultiTurnGRPOWorkerFactory`**: Factory for creating domain-specific workers

## Configuration

### Base Configuration (`multiturn_grpo_trainer.yaml`)

```yaml
algorithm:
  adv_estimator: grpo_multiturn
  multiturn_grpo:
    n_trajectories_per_question: 4
    max_turns_per_trajectory: 5
    max_context_length: 2048

multiturn_trajectory:
  context_condenser:
    type: simple
    keep_last_n_steps: 2
  generation:
    max_sequence_length: 2048
    batch_size: 32

reward_computation:
  type: grpo
  domain: general
```

### Math-Specific Configuration (`multiturn_grpo_math.yaml`)

Specialized settings for mathematical reasoning:
- More trajectories per question (6)
- Longer context for mathematical derivations
- Math-specific format validation
- Lower temperature for focused reasoning

## Usage Examples

### Basic Usage

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=multiturn_grpo_trainer \
    algorithm.adv_estimator=grpo_multiturn \
    data.train_files="['path/to/train.parquet']" \
    data.val_files="['path/to/val.parquet']"
```

### Math Problems

```bash
./examples/multiturn_grpo_trainer/run_multiturn_grpo_math.sh
```

### Custom Configuration

```python
from verl.workers.rollout.multi_turn_grpo_worker import MultiTurnGRPOWorkerFactory

# Create a math-specialized worker
worker = MultiTurnGRPOWorkerFactory.create_math_worker(
    base_rollout=your_rollout,
    tokenizer=your_tokenizer,
    n_trajectories=6,
    max_turns=7
)

# Generate trajectories and compute advantages
results = worker.rollout_and_compute_advantages(prompts)
```

## Key Parameters

### Trajectory Generation
- `n_trajectories_per_question`: Number of trajectories to generate per question (GRPO group size)
- `max_turns_per_trajectory`: Maximum number of turns in each trajectory
- `max_context_length`: Maximum context length for condensation

### Context Condensation
- `keep_last_n_steps`: Number of recent steps to keep in condensed context
- `max_condensed_tokens`: Maximum tokens in condensed context

### Reward Computation
- `domain`: Problem domain ("general", "math", "coding")
- `ground_truth_file`: Path to ground truth answers for evaluation
- `format_validation`: Rules for validating trajectory format

### GRPO Settings
- `norm_adv_by_std_in_grpo`: Whether to normalize advantages by standard deviation
- `advantage_epsilon`: Numerical stability parameter

## Reward Logic

The trajectory reward follows this logic:

1. **Format Validation**: Check if trajectory has valid format
   - Non-empty responses
   - Proper structure indicators
   - Domain-specific requirements (e.g., math keywords, code blocks)

2. **Answer Correctness**: Validate the final answer
   - Compare with ground truth (if available)
   - Heuristic validation (answer indicators, numeric content)
   - Domain-specific validation

3. **Context Efficiency**: Reward shorter contexts
   - `reward = 1 / max_context_length` for correct answers
   - Encourages efficient reasoning

## Integration with Existing VERL

The multi-turn GRPO implementation integrates seamlessly with existing VERL components:

- **Rollout Workers**: Extends existing rollout implementations (VLLM, SGLang)
- **Training Pipeline**: Uses standard PPO training loop with custom advantage estimator
- **Configuration System**: Leverages Hydra configuration management
- **Distributed Training**: Compatible with Ray-based distributed training

## Performance Considerations

1. **Memory Usage**: Multi-turn trajectories require more memory
   - Use gradient checkpointing for longer sequences
   - Consider smaller batch sizes

2. **Computational Cost**: Multiple trajectories per question increase compute
   - Adjust `n_trajectories_per_question` based on resources
   - Use efficient rollout backends (VLLM, SGLang)

3. **Context Management**: Effective condensation is crucial
   - Balance between context preservation and efficiency
   - Domain-specific condensation strategies

## Future Extensions

1. **Advanced Context Condensation**: Attention-based, summary-based condensers
2. **Dynamic Trajectory Length**: Adaptive termination based on confidence
3. **Hierarchical Rewards**: Step-level and trajectory-level reward combination
4. **Multi-Modal Support**: Extension to vision-language tasks

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or sequence length
2. **Slow Training**: Check trajectory generation efficiency
3. **Poor Convergence**: Adjust reward computation or advantage normalization

### Debug Mode

Enable trajectory profiling for debugging:

```yaml
trajectory_profiling:
  enable: true
  profile_reward_computation: true
  profile_context_condensation: true
```

## Citation

If you use this multi-turn GRPO implementation, please cite:

```bibtex
@misc{verl_multiturn_grpo,
  title={Multi-Turn GRPO with Context Condensation},
  author={VERL Team},
  year={2024},
  url={https://github.com/volcengine/verl}
}
```