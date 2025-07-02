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

## Evaluation and Testing

### 1. Comprehensive Model Evaluation

After training, evaluate your model's performance against the baseline:

```bash
cd examples/arc_vision

# Evaluate on test set
python evaluate_arc_vision.py \
    --model_path ~/verl/outputs/arc_vision/[experiment]/checkpoint-[step] \
    --split test \
    --max_samples 200

# Full evaluation (all samples)
python evaluate_arc_vision.py \
    --model_path ~/verl/outputs/arc_vision/[experiment]/checkpoint-[step] \
    --split test
```

This will output:
- **Primary metrics**: Accuracy improvement, IoU scores, tool precision
- **Tool patterns**: Which tools are used for which UI challenges
- **Detailed analysis**: JSON file with per-sample results

### 2. Generate Training Report

Create comprehensive visualizations and analysis of your training run:

```bash
# Generate report from training logs
python generate_training_report.py \
    --log_dir ~/verl/outputs/arc_vision/[experiment] \
    --output_dir ./training_reports

# View results
open training_reports/training_report_*/report.html
```

The report includes:
- Training progress plots (accuracy, IoU, tool usage)
- Reward component analysis
- Performance summary for research paper
- Markdown file with values to replace paper placeholders

### 3. Interactive Model Testing

Test your trained model on custom images:

```bash
# Interactive testing session
python test_model_inference.py \
    --model_path ~/verl/outputs/arc_vision/[experiment]/checkpoint-[step]

# Batch testing with JSON file
python test_model_inference.py \
    --model_path ~/verl/outputs/arc_vision/[experiment]/checkpoint-[step] \
    --batch_test test_cases.json

# Create example test file
python test_model_inference.py --create_example
```

Features:
- Test on any image (local file or URL)
- Visualize bounding box predictions
- See tool usage and confidence changes
- Save results for analysis

## Example Usage

### Complete Training and Evaluation Pipeline

```bash
# 1. Prepare data
cd examples/arc_vision
python prepare_screenspot_data.py

# 2. Train model
bash run_arc_vision_3b.sh

# 3. Monitor training
tensorboard --logdir ~/verl/outputs/arc_vision

# 4. Generate training report
python generate_training_report.py \
    --log_dir ~/verl/outputs/arc_vision/latest

# 5. Evaluate on test set
python evaluate_arc_vision.py \
    --model_path ~/verl/outputs/arc_vision/latest/checkpoint-final

# 6. Test interactively
python test_model_inference.py \
    --model_path ~/verl/outputs/arc_vision/latest/checkpoint-final
```

### Testing on Custom UI Screenshot

```python
# In interactive mode
Enter image path/URL or command: /path/to/screenshot.png
What should I find in this image? Find the submit button

# Model will show:
# 1. Reasoning about the UI element
# 2. Whether tools are needed
# 3. Bounding box visualization
```

## Research Results

After training and evaluation, update the research paper placeholders:

| Placeholder | Replace With | Found In |
|-------------|--------------|----------|
| [X]% | Accuracy improvement | evaluate_arc_vision.py output |
| [Y]% | False positive reduction | evaluate_arc_vision.py output |
| [P]% | Tool precision | evaluate_arc_vision.py output |

The `generate_training_report.py` script creates a `results_for_paper.md` file with all values pre-formatted for the research paper.

## Citation

This work is part of Arc's vision for continuous AI systems. For more details, see the [research paper](../../research/blog_post_vision_rl_final.md).