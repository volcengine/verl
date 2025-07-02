# Arc Vision Reward Function Implementation

## Overview

This document describes the complete implementation of the Arc Vision reward function as specified in the blog post research vision. The reward function uses a 3-component approach to guide the model's learning of when and how to use tools effectively.

## Reward Function Formula

The reward function is defined as:

```
R(s,a,t) = α*R_task + β*R_tool + γ*R_gate
```

Where:
- **α = 0.6**: Weight for task performance
- **β = 0.3**: Weight for tool effectiveness  
- **γ = 0.1**: Weight for gating penalties
- **τ = 0.7**: Confidence threshold for tool use

## Component Details

### 1. R_task: IoU-based Detection Accuracy

The task reward measures how accurately the model detects UI elements:

```python
R_task = IoU(predicted_bbox, ground_truth_bbox)
```

- Extracts predicted bounding box from `<bbox>[x1, y1, x2, y2]</bbox>` tags
- Calculates Intersection over Union (IoU) with ground truth
- Range: [0, 1] where 1 is perfect detection

### 2. R_tool: Tool Effectiveness Based on Confidence Gain

The tool reward incentivizes effective tool usage:

```python
if tools_used:
    confidence_gain = confidence_after - confidence_before
    R_tool = max(0.0, confidence_gain * IoU)
else:
    R_tool = 0.0
```

- Measures confidence improvement from tool use
- Weighted by IoU to ensure tools lead to correct results
- Range: [0, 1] where higher values indicate effective tool use

### 3. R_gate: Penalties for Tool Misuse

The gating component penalizes inappropriate tool usage:

```python
R_gate = sum(penalties) where penalties include:
- unnecessary_tool: -0.5 (high confidence but used tools)
- missed_opportunity: -0.3 (low confidence but no tools)
- ineffective_tool: -0.2 (tools used but no confidence gain)
- excessive_tools: -0.4 (more than 3 tool invocations)
```

## Confidence Estimation

Since explicit confidence scores aren't always available, we use multiple signals:

1. **Tool Usage as Proxy**: Tool invocation implies lower confidence
2. **Reasoning Analysis**: Searches for uncertainty phrases like "unclear", "difficult to see"
3. **Stated Confidence**: Extracts explicit confidence if mentioned

### Confidence Calculation:
```python
confidence_before = min(tool_confidence, reasoning_confidence)
confidence_after = confidence_before + 0.15 * num_tools_used
```

## Tool Usage Parsing

The implementation extracts tool usage from response tags:

```python
<tool_call>
    <name>zoom_ui_element</name>
    <parameters>...</parameters>
</tool_call>
```

Supported tools:
- `zoom_ui_element`: Zooms into specific UI regions
- `wait_for_ui`: Waits for UI state changes
- `inspect_element`: Gets detailed element information

## Detailed Logging

The reward function includes comprehensive logging for analysis:

1. **Reasoning Traces** (`reasoning_traces.jsonl`): Captures model reasoning patterns
2. **Confidence Calibration** (`confidence_calibration.jsonl`): Tracks confidence accuracy
3. **Tool Patterns** (`tool_patterns.jsonl`): Analyzes tool effectiveness
4. **Contradictions** (`contradictions.jsonl`): Identifies reasoning inconsistencies

## Usage Example

```python
from arc_vision_custom_reward import arc_vision_compute_score_fn

# Create reward function with custom parameters
compute_score = arc_vision_compute_score_fn(
    confidence_threshold=0.7,
    reward_weights={"task": 0.6, "tool": 0.3, "gate": 0.1},
    tool_penalties={
        "unnecessary_tool": -0.5,
        "missed_opportunity": -0.3,
        "ineffective_tool": -0.2,
        "excessive_tools": -0.4
    }
)

# Compute reward
result = compute_score(
    data_source="arc_vision",
    solution_str=model_response,
    ground_truth=[100, 200, 300, 400]  # [x1, y1, x2, y2]
)

# Access detailed metrics
print(f"Total Reward: {result['reward']}")
print(f"Task Component: {result['r_task']}")
print(f"Tool Component: {result['r_tool']}")
print(f"Gate Component: {result['r_gate']}")
print(f"IoU: {result['iou']}")
print(f"Confidence: {result['confidence_before']} -> {result['confidence_after']}")
```

## Key Design Decisions

1. **Tool Reward Coupling**: R_tool is multiplied by IoU to ensure tools only get rewarded when they lead to correct results

2. **Confidence Threshold**: τ = 0.7 balances between over-cautious tool use and missing opportunities

3. **Penalty Magnitudes**: Penalties are calibrated to discourage misuse without overwhelming the positive rewards

4. **Minimum Reward**: Final reward is clamped to [0, ∞) to maintain training stability

## Integration with VERL

The reward function follows VERL's `compute_score` interface:
- Takes `data_source`, `solution_str`, `ground_truth` as required parameters
- Returns a dictionary with at least a `reward` key
- Additional metrics provided for analysis and debugging

## Future Enhancements

1. **Learnable Confidence**: Train a separate head to predict confidence explicitly
2. **Dynamic Thresholds**: Adjust τ based on task difficulty or element type
3. **Tool-Specific Rewards**: Different rewards for different tool types
4. **Multi-Step Rewards**: Consider sequences of tool invocations