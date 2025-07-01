# Arc Vision RL → VERL Integration Guide

## Project Context

### What We're Building
Arc Vision RL is a confidence-gated tool learning framework that teaches vision-language models (VLMs) to recognize when they need help and autonomously invoke tools to improve their performance. This addresses a critical gap in current VLMs: while they achieve 85.71% accuracy on tool recognition tasks, they score only 58.79% on the mechanical reasoning that underlies effective tool use.

### The Problem
Our baseline evaluation reveals that Qwen2.5-VL-3B achieves only **0.5% accuracy** on UI element detection tasks (ScreenSpot benchmark), with 86.7% of attempts yielding zero IoU overlap. This extreme failure rate demonstrates that current VLMs lack the visual grounding necessary for reliable automation.

### Our Solution
We use reinforcement learning (specifically Group Relative Policy Optimization - GRPO) to teach models:
1. **When to use tools** - Through confidence-based gating (threshold τ = 0.7)
2. **Which tools help** - Learning tool selection strategies without human supervision
3. **When NOT to use tools** - Preventing unnecessary tool invocations that waste compute

### Key Innovation: 3-Component Reward Structure
```
R(s, a, t) = α R_task(s, a) + β R_tool(s, a, t) + γ R_gate(c, t)
```
- **R_task**: Task performance (IoU for UI detection)
- **R_tool**: Tool effectiveness (confidence gain after tool use)
- **R_gate**: Gating penalty (prevents tool abuse)

### Why This Matters
This work is part of Arc's broader vision for continuous AI systems that learn from production failures. UI automation is our strategic starting point because:
- **Discrete and Measurable**: Clear success metrics (IoU > 0.5)
- **High-Frequency Failures**: Dense feedback signal for RL
- **Production Data**: Real failures, not synthetic benchmarks
- **Foundation for Scale**: Patterns transfer to document understanding, data extraction, and beyond

### Integration Goal
We need to fork VERL (a proven GRPO implementation) and add:
1. Vision-language model support (Qwen2.5-VL-3B)
2. Confidence extraction from generation scores
3. Tool invocation pipeline (zoom, wait, inspect)
4. Composite reward computation
5. ScreenSpot dataset integration

This document contains all the essential components from Arc Vision RL that need to be integrated into a forked VERL implementation to achieve confidence-gated tool learning for vision-language models.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Key Components to Port](#key-components-to-port)
3. [Data Format Requirements](#data-format-requirements)
4. [Reward System Implementation](#reward-system-implementation)
5. [Tool Learning Integration](#tool-learning-integration)
6. [Configuration Parameters](#configuration-parameters)
7. [Modified VERL Training Pipeline](#modified-verl-training-pipeline)

## Core Concepts

### Confidence-Gated Tool Learning
Our approach teaches VLMs to recognize when they need tools through a two-stage process:

```python
# Stage 1: Initial detection attempt
a₁ = π(s, θ)  # Standard VLM output
c₁ = confidence(a₁)  # Extract confidence from softmax over detection logits

# Stage 2: Conditional tool invocation
if c₁ < τ:  # Confidence threshold τ = 0.7
    t = π_tool(s, a₁, θ)  # Select tool
    s' = apply_tool(s, t)  # Enhanced state
    a₂ = π(s', θ)  # Re-attempt with tool
```

### Key Innovation
- **Baseline Performance**: Qwen2.5-VL-3B achieves only 0.5% accuracy on UI detection
- **Goal**: Learn when tools help (not just how to use them)
- **Method**: 3-component reward structure with GRPO

## Key Components to Port

### 1. Dataset Integration (`src/data/dataset.py`)
```python
from datasets import load_dataset
from PIL import Image
import numpy as np

class ScreenSpotDataset:
    """ScreenSpot benchmark for UI element detection."""
    
    def __init__(self, split="train", max_samples=None):
        self.dataset = load_dataset("ScreenSpot/screenspot", "desktop", split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "image": sample["image"],  # PIL Image
            "instruction": sample["instruction"],  # UI element description
            "bbox": np.array(sample["bbox"], dtype=np.float32)  # [x1, y1, x2, y2]
        }
```

### 2. Tool System (`src/tools.py`)
```python
from PIL import Image
import numpy as np

def apply_zoom_tool(image: Image.Image, region: list, zoom_factor: float = 2.0) -> Image.Image:
    """Apply zoom tool to enhance UI element visibility."""
    x1, y1, x2, y2 = region
    
    # Convert normalized to pixel coordinates
    width, height = image.size
    x1_px = int(x1 * width)
    y1_px = int(y1 * height)
    x2_px = int(x2 * width)
    y2_px = int(y2 * height)
    
    # Crop and resize
    cropped = image.crop((x1_px, y1_px, x2_px, y2_px))
    new_size = (int(cropped.width * zoom_factor), int(cropped.height * zoom_factor))
    zoomed = cropped.resize(new_size, Image.LANCZOS)
    
    return zoomed

TOOL_REGISTRY = {
    "zoom": apply_zoom_tool,
    "wait": lambda img, _: img,  # Placeholder
    "inspect": lambda img, _: img  # Placeholder
}
```

### 3. Composite Reward Function (`src/training/rewards.py`)
```python
import numpy as np

def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_arc_vision_reward(
    trajectory,
    ground_truth,
    confidence_threshold: float = 0.7,
    weights: dict = {"base": 0.6, "tool": 0.3, "gate": 0.1}
) -> float:
    """
    Compute 3-component reward for Arc Vision RL.
    
    Components:
    1. Base reward (IoU)
    2. Tool effectiveness (confidence gain)
    3. Gating penalty (prevent tool abuse)
    """
    # Task performance (IoU-based)
    r_task = compute_iou(trajectory.bbox, ground_truth.bbox)
    
    # Tool effectiveness (confidence-based)
    r_tool = 0.0
    if trajectory.tool_used:
        δ_conf = trajectory.conf_after - trajectory.conf_before
        if trajectory.conf_before < confidence_threshold and trajectory.conf_after >= confidence_threshold:
            r_tool = 1.0  # Crossed threshold
        elif δ_conf > 0:
            r_tool = δ_conf * 2.0  # Positive gain
        else:
            r_tool = -0.2  # Tool didn't help
    
    # Gating penalty (prevent reward hacking)
    r_gate = 0.0
    if trajectory.conf_before > confidence_threshold and trajectory.tool_used:
        r_gate = -0.5  # Unnecessary tool use
    elif trajectory.conf_before < confidence_threshold and not trajectory.tool_used:
        r_gate = -0.3  # Missed opportunity
    
    return weights["base"] * r_task + weights["tool"] * r_tool + weights["gate"] * r_gate
```

### 4. Confidence Extraction
```python
def extract_confidence_from_generation(outputs, scores):
    """Extract confidence from model generation scores."""
    if not scores:
        return 0.5  # Default
    
    # Calculate entropy-based confidence
    entropies = []
    token_probs = []
    
    for score in scores:
        probs = torch.softmax(score, dim=-1)
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
        entropies.append(entropy)
        
        # Max probability for each token
        max_prob = probs.max(dim=-1).values.mean().item()
        token_probs.append(max_prob)
    
    # Convert entropy to confidence
    avg_entropy = np.mean(entropies)
    max_entropy = np.log(score.shape[-1])  # Vocabulary size
    entropy_confidence = 1 - (avg_entropy / max_entropy)
    
    # Combine measures
    prob_confidence = np.mean(token_probs)
    confidence = 0.7 * prob_confidence + 0.3 * entropy_confidence
    
    return confidence
```

## Data Format Requirements

### VERL Parquet Format
```python
import pandas as pd
import json

def convert_to_verl_format(dataset):
    """Convert Arc Vision data to VERL-compatible format."""
    records = []
    
    for idx, sample in enumerate(dataset):
        # Create reasoning-enhanced prompt
        prompt = f"""{sample["instruction"]}

First, analyze the image and describe what you observe about the target element:
<reasoning>
- Is the element clearly visible or partially obscured?
- Is it small, blurry, or low contrast?
- What challenges do you face in locating it?
- Do you need to use tools to see it better?
</reasoning>

Then provide the bounding box coordinates [x1, y1, x2, y2]."""
        
        record = {
            "prompt": prompt,
            "images": [save_image(sample["image"], idx)],  # Save and return path
            "ground_truth": json.dumps(sample["bbox"].tolist()),
            "metadata": json.dumps({
                "sample_idx": idx,
                "original_instruction": sample["instruction"]
            })
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet("train.parquet")
    return "train.parquet"
```

## Reward System Implementation

### Custom VERL Reward Module
```python
# File: verl_arc_vision/reward_model.py

class ArcVisionRewardModel:
    """VERL-compatible reward model for Arc Vision RL."""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.weights = {"base": 0.6, "tool": 0.3, "gate": 0.1}
    
    def __call__(self, prompts, responses, ground_truths, **kwargs):
        """Compute rewards for VERL training."""
        rewards = []
        
        for response, gt_str in zip(responses, ground_truths):
            # Parse response
            trajectory = self.parse_response(response)
            gt_bbox = json.loads(gt_str)
            
            # Compute composite reward
            reward = compute_arc_vision_reward(
                trajectory, 
                gt_bbox,
                self.confidence_threshold,
                self.weights
            )
            rewards.append(reward)
        
        return rewards
    
    def parse_response(self, response):
        """Parse model response for bbox, tool use, and confidence."""
        # Extract bbox: [x1, y1, x2, y2]
        bbox_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]'
        match = re.search(bbox_pattern, response)
        bbox = [float(x) for x in match.groups()] if match else [0, 0, 0, 0]
        
        # Check tool use
        tool_used = "zoom" in response.lower() and "<tool>" in response
        
        # Parse confidence (if included in response)
        conf_pattern = r'confidence:\s*(\d+\.?\d*)'
        conf_match = re.search(conf_pattern, response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        return SimpleNamespace(
            bbox=np.array(bbox),
            tool_used=tool_used,
            conf_before=confidence * 0.8 if tool_used else confidence,
            conf_after=confidence
        )
```

## Tool Learning Integration

### Modified Generation Pipeline
```python
# File: verl_arc_vision/tool_augmented_rollout.py

def generate_with_tool_learning(model, processor, inputs, confidence_threshold=0.7):
    """Generate with confidence-gated tool learning."""
    
    # Stage 1: Initial generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    # Extract confidence
    confidence = extract_confidence_from_generation(outputs, outputs.scores)
    
    # Stage 2: Tool invocation if low confidence
    if confidence < confidence_threshold:
        # Parse initial output for region
        initial_text = processor.decode(outputs.sequences[0])
        region = extract_region_from_text(initial_text)
        
        if region:
            # Apply zoom tool
            enhanced_image = apply_zoom_tool(inputs.image, region)
            
            # Re-generate with enhanced image
            enhanced_inputs = processor(
                text=inputs.text + "\n<tool>zoom applied</tool>",
                images=[enhanced_image],
                return_tensors="pt"
            )
            
            enhanced_outputs = model.generate(
                **enhanced_inputs,
                max_new_tokens=100,
                temperature=0.5  # Lower temp for tool-enhanced
            )
            
            return {
                "text": processor.decode(enhanced_outputs.sequences[0]),
                "tool_used": "zoom",
                "confidence_before": confidence,
                "confidence_after": extract_confidence_from_generation(enhanced_outputs)
            }
    
    return {
        "text": processor.decode(outputs.sequences[0]),
        "tool_used": None,
        "confidence": confidence
    }
```

## Configuration Parameters

### VERL Training Configuration
```yaml
# File: verl_arc_vision/config.yaml

# Model configuration
model:
  path: "Qwen/Qwen2.5-VL-3B-Instruct"
  torch_dtype: bfloat16
  trust_remote_code: true
  use_gradient_checkpointing: true

# Data configuration  
data:
  train_files: "data/verl_format/train.parquet"
  train_batch_size: 512
  max_prompt_length: 512
  max_response_length: 256
  image_key: "images"

# GRPO configuration
algorithm:
  adv_estimator: "grpo"
  gamma: 1.0
  lam: 1.0
  adv_norm: true

# Actor configuration
actor:
  optim:
    lr: 1e-6
    weight_decay: 0.01
  ppo_epochs: 1
  ppo_mini_batch_size: 256
  ppo_micro_batch_size_per_gpu: 2
  gradient_accumulation_steps: 16
  use_kl_loss: true
  kl_loss_coef: 0.04
  entropy_loss_coef: 0.01  # Prevent mode collapse

# Rollout configuration
rollout:
  n: 5  # Number of candidates per prompt
  temperature_range: [0.3, 0.5, 0.7, 0.9, 1.0]
  gpu_memory_utilization: 0.6

# Arc Vision specific
arc_vision:
  confidence_threshold: 0.7
  enable_tools: true
  tool_penalty: 0.5
  reward_weights:
    base: 0.6
    tool: 0.3
    gate: 0.1
```

## Modified VERL Training Pipeline

### Launch Script
```bash
#!/bin/bash
# File: launch_arc_vision_verl.sh

# Fork and setup
git clone https://github.com/volcengine/verl verl-arc-vision
cd verl-arc-vision

# Add Arc Vision components
cp -r ../arc-rl/src/data ./verl/data/arc_vision
cp -r ../arc-rl/src/tools.py ./verl/tools/
cp -r ../arc-rl/src/training/rewards.py ./verl/reward_models/

# Prepare data
python prepare_arc_vision_data.py \
    --dataset screenspot \
    --output data/verl_format/

# Launch training
python -m verl.trainer.main_ppo \
    --config configs/arc_vision_grpo.yaml \
    --reward_model verl.reward_models.arc_vision:ArcVisionRewardModel \
    --rollout_generator verl.rollout.arc_vision:ToolAugmentedRollout \
    data.train_files=data/verl_format/train.parquet \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.project_name=arc_vision_rl \
    trainer.experiment_name=confidence_gated_tool_learning \
    trainer.default_local_dir=outputs/verl_arc_vision
```

### Integration Points in VERL

1. **Custom Reward Model**: Replace VERL's default reward with `ArcVisionRewardModel`
2. **Rollout Generator**: Modify to include tool-augmented generation
3. **Data Loader**: Extend to handle image+text inputs
4. **Metrics**: Add tool usage tracking and confidence monitoring

### Key Modifications to VERL Core

```python
# In verl/trainer/ppo_trainer.py (pseudo-code)

class PPOTrainer:
    def generate_rollouts(self, prompts):
        # Add Arc Vision tool learning
        if self.config.get("arc_vision.enable_tools"):
            return self.tool_augmented_rollout(prompts)
        else:
            return self.standard_rollout(prompts)
    
    def compute_rewards(self, trajectories):
        # Use Arc Vision composite rewards
        if hasattr(self.reward_model, "compute_arc_vision_reward"):
            return self.reward_model(trajectories)
        else:
            return self.default_rewards(trajectories)
```

## Quick Start

1. Fork VERL repository
2. Copy Arc Vision components to appropriate directories
3. Install additional dependencies:
   ```bash
   pip install datasets pillow pandas
   ```
4. Run data preparation script
5. Launch training with modified configuration

## Expected Results

Based on the blog post targets and baseline evaluation:

### Primary Metrics
- **Accuracy Improvement**: From 0.5% baseline to [X]% on UI detection (IoU > 0.5)
- **Tool Precision**: [P]% of tool invocations should improve prediction accuracy
- **False Positive Reduction**: [Y]% reduction in unnecessary tool use

### Secondary Metrics  
- **Detection Rate**: Maintain 98% (model outputs valid bounding boxes)
- **Average IoU**: Improve from baseline 0.026 to meaningful overlap
- **Zero IoU Reduction**: Decrease from 86.7% complete failures
- **Confidence Calibration**: Model confidence should correlate with actual performance

### Why These Metrics Matter
- **For Engineering Teams**: Reduced test maintenance from reliable UI detection
- **For Production**: Lower computational costs through strategic tool use
- **For Arc's Vision**: Proof that production failures can drive autonomous improvement

## Monitoring Progress

Track these metrics during training:
- `arc_vision/accuracy`: Overall detection accuracy
- `arc_vision/tool_usage_rate`: Percentage of samples using tools
- `arc_vision/tool_precision`: Success rate when tools are used
- `arc_vision/confidence_calibration`: How well confidence predicts success

## Troubleshooting

1. **Memory Issues**: Reduce `rollout.n` or `data.train_batch_size`
2. **Slow Training**: Enable `model.use_gradient_checkpointing`
3. **Poor Tool Learning**: Adjust `arc_vision.reward_weights`
4. **Mode Collapse**: Increase `actor.entropy_loss_coef`

---

This guide contains all essential components from Arc Vision RL needed to integrate with VERL. The key insight is that we're not reimplementing GRPO - we're adding vision support, tool learning, and composite rewards to VERL's proven infrastructure.

## Research Context

### Core Research Question
Can we teach compact vision-language models to recognize their own limitations and strategically invoke tools, achieving performance comparable to much larger models?

### Key Findings from Literature
- **Core Knowledge Deficit** (Li et al., 2025): MLMs achieve 85.71% on tool recognition but only 58.79% on mechanical reasoning
- **Online vs Offline RL** (Lanchantin et al., 2025): Online GRPO significantly outperforms offline methods for verifiable tasks
- **Tool Gating** (Kumar et al., 2025): Tool rewards must be "gated" to prevent abuse
- **Entropy Collapse** (Multiple sources): Critical challenge in verifiable tasks - requires entropy regularization

### Our Contribution
1. **Confidence-Gated Learning**: First to decompose tool learning into confidence estimation + conditional selection
2. **Production-Driven RL**: Learning from real UI failures, not synthetic benchmarks  
3. **Composite Rewards**: Balancing task performance, tool effectiveness, and computational efficiency

### Broader Impact
This work validates Arc's thesis that production failures are not problems to be managed, but data to be learned from. The principles established here - confidence-based gating, production-driven learning, and tool-aware rewards - will scale to document understanding, data extraction, and the full taxonomy of agent failures.

---

*For the complete research paper and findings, see: `/research/blog_post_vision_rl_final.md`*