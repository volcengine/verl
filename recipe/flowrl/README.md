# FlowRL: Flow Matching for Reinforcement Learning

FlowRL is a novel reinforcement learning algorithm that leverages flow matching techniques for policy optimization. This implementation provides a clean integration with VERL without modifying the core framework.

## Key Features

- **Flow Matching Integration**: Uses partition function Z for improved policy optimization
- **Trajectory Balance Loss**: Replaces traditional PPO loss with FlowRL objective
- **Clean Architecture**: Implements FlowRL as a recipe without modifying VERL core code
- **Efficient Training**: Maintains compatibility with existing VERL training infrastructure

## Algorithm Overview

FlowRL modifies the standard RL training pipeline in four key ways:

1. **Partition Function Z**: Adds a projection module `ProjZModule` to estimate log partition function
2. **Enhanced Forward Pass**: Modifies actor forward pass to return log Z values
3. **FlowRL Objective**: Replaces PPO loss with trajectory balance (TB) loss
4. **Parameter Management**: Handles proj_z parameters separately during model loading

## Quick Start

```bash
# Basic FlowRL training
bash recipe/flowrl/run_flowrl_qwen.sh

# Custom configuration
python recipe/flowrl/main_flowrl.py \
    --config recipe/flowrl/config/flowrl_config.yaml \
    --model.path "your-model-path"
```

## Configuration

Key FlowRL-specific parameters:

- `actor.proj_layer`: Number of layers in projection network (default: 3)
- `actor.proj_dropout`: Dropout rate for projection network (default: 0.1)
- `algorithm.tb_coef`: Trajectory balance loss coefficient (default: 15.0)
- `algorithm.importance_sampling`: Enable importance sampling (default: True)

## Reference

If you use FlowRL in your research, please cite:

```bibtex
@article{flowrl2024,
  title={FlowRL: Flow Matching for Reinforcement Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Implementation Details

This recipe implements FlowRL by:

1. Extending the standard actor with a projection network for log Z estimation
2. Implementing a custom FlowRL trainer that uses trajectory balance loss
3. Providing configuration templates for different model sizes
4. Maintaining full compatibility with VERL's distributed training infrastructure

For detailed implementation, see the source files in this directory.