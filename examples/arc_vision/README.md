# Arc Vision RL: Confidence-Gated Tool Learning for UI Detection

This example demonstrates how to train vision-language models to autonomously invoke tools for UI element detection using VERL's GRPO implementation.

## Overview

Arc Vision RL teaches Qwen2.5-VL-3B to:
1. **Recognize when it needs help** - Through confidence-based gating (τ = 0.7)
2. **Select appropriate tools** - Zoom, wait, or inspect based on the challenge
3. **Avoid tool abuse** - Learn when NOT to use tools

Starting from a baseline of only 0.5% accuracy on ScreenSpot UI detection, our approach uses a 3-component reward structure to enable strategic tool use.

## Quick Start

### 1. Prepare ScreenSpot Dataset

```bash
cd examples/arc_vision
python prepare_screenspot_data.py --max_samples 1000  # Use full dataset in production
```

This downloads the official ScreenSpot dataset from [Hugging Face](https://huggingface.co/datasets/rootsautomation/ScreenSpot) and converts it to VERL format.

### 2. Run Training

```bash
# From VERL root directory
bash examples/arc_vision/run_arc_vision_3b.sh
```

## Architecture

### Tools
- **zoom_ui_element**: Enhances small or unclear UI elements
- **wait_for_ui**: Handles loading states and animations  
- **inspect_element**: Provides structural information about UI

### Reward Components
1. **Task Reward (60%)**: IoU between predicted and ground truth bbox
2. **Tool Reward (30%)**: Confidence gain from tool use
3. **Gating Penalty (10%)**: Prevents unnecessary tool invocations

### Key Innovation
The model learns to use tools based on its own confidence, not fixed rules. This enables:
- Efficient tool use only when beneficial
- Adaptation to different UI patterns
- Continuous improvement from production failures

## Configuration

Key parameters in `config/arc_vision_grpo.yaml`:
```yaml
arc_vision:
  confidence_threshold: 0.7  # Tool invocation threshold
  reward_weights:
    task: 0.6
    tool: 0.3
    gate: 0.1
```

## Expected Results

From the baseline evaluation:
- **Baseline Accuracy**: 0.5% (IoU > 0.5)
- **Detection Rate**: 98% (valid bbox format)
- **Average IoU**: 0.026
- **Zero IoU Rate**: 86.7%

With Arc Vision RL:
- **Target Accuracy**: >30% improvement
- **Tool Precision**: >70% of tool uses should improve accuracy
- **Reduced False Positives**: <20% unnecessary tool invocations

## Monitoring

Track training progress with:
```bash
tensorboard --logdir outputs/arc_vision/
```

Key metrics:
- `arc_vision/accuracy`: Overall detection accuracy
- `arc_vision/tool_usage_rate`: Percentage using tools
- `arc_vision/tool_precision`: Tool effectiveness
- `arc_vision/confidence_calibration`: Confidence vs performance

## Troubleshooting

### Memory Issues
```yaml
# Reduce batch size
data.train_batch_size: 256  # Default: 512

# Enable gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing: true
```

### Poor Tool Learning
Adjust reward weights:
```yaml
arc_vision.reward_weights:
  task: 0.5   # Reduce if model ignores tools
  tool: 0.4   # Increase to encourage tool exploration
  gate: 0.1
```

### Entropy Collapse
```yaml
actor_rollout_ref.actor.entropy_loss_coef: 0.02  # Increase from 0.01
```

## Citation

This work is part of Arc's vision for continuous AI systems. For more details, see the [research paper](../../research/blog_post_vision_rl_final.md).