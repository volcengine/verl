# Arc Vision Reward Function Requirements

## Critical Implementation Details

This document ensures future modifications maintain alignment with the Arc Vision research goals from `research/blog_post_vision_rl_final.md`.

## 1. Core Formula (MUST IMPLEMENT ALL THREE COMPONENTS)

```
R(s,a,t) = α*R_task + β*R_tool + γ*R_gate
```

### Component Breakdown:

#### R_task: Task Performance (IoU-based)
- **What**: Intersection over Union between predicted and ground truth bboxes
- **Range**: [0, 1]
- **Implementation**: Standard IoU calculation
- **Weight**: α = 0.6

#### R_tool: Tool Effectiveness 
- **What**: Rewards successful tool usage that improves confidence
- **Formula**: `confidence_gain * success_factor`
- **Key Logic**:
  ```python
  if tool_used and task_successful:
      R_tool = max(0, confidence_after - confidence_before)
  else:
      R_tool = 0
  ```
- **Weight**: β = 0.3

#### R_gate: Gating Penalties
- **What**: Penalties for incorrect tool decisions
- **Four penalty types**:
  1. **Unnecessary tool** (-0.5): High confidence (>0.7) but used tools anyway
  2. **Missed opportunity** (-0.3): Low confidence (<0.7), no tools, poor result
  3. **Ineffective tool** (-0.2): Used tools but no confidence gain
  4. **Excessive tools** (-0.4): More than 3 tool invocations
- **Weight**: γ = 0.1

## 2. Confidence Extraction

Since models don't always output explicit confidence, use multiple signals:

1. **Direct statements**: "I'm 90% confident" → 0.9
2. **Tool usage as proxy**: No tools → high confidence, tools → low confidence
3. **Reasoning analysis**: Look for uncertainty phrases
4. **Default assumptions**: 
   - Before tools: 0.4 (low)
   - After tools: 0.8 (high)

## 3. Tool Usage Parsing

Extract from response:
```xml
<tool_call>
{"name": "zoom_ui_element", "arguments": {...}}
</tool_call>
```

Track:
- Tool count
- Tool types (zoom, wait, inspect)
- Order of invocation

## 4. Critical Parameters

```python
confidence_threshold = 0.7  # τ in the paper
reward_weights = {
    "task": 0.6,    # α
    "tool": 0.3,    # β  
    "gate": 0.1     # γ
}
```

## 5. Return Format

VERL expects:
```python
return {"reward": float(total_reward)}
```

## 6. Common Mistakes to Avoid

### ❌ DON'T: Implement only R_task
```python
# WRONG - Missing tool and gate components
reward = iou * 0.6
```

### ❌ DON'T: Use wrapper functions
```python
# WRONG - Causes keyword argument conflicts
def wrapper(**kwargs):
    return real_function(param=value, **kwargs)
```

### ❌ DON'T: Return complex dictionaries
```python
# WRONG - VERL expects {"reward": float}
return {"score": x, "confidence": y, "details": z}
```

### ✅ DO: Implement all three components
```python
# CORRECT
reward = (
    weights["task"] * iou +
    weights["tool"] * tool_effectiveness +
    weights["gate"] * penalties
)
return {"reward": reward}
```

## 7. Testing Requirements

Any changes MUST pass these tests:

1. **Perfect detection without tools**: High R_task, zero R_tool, no penalties
2. **Effective tool use**: Moderate R_task, positive R_tool, no penalties
3. **Unnecessary tool penalty**: High confidence but used tools
4. **Missed opportunity penalty**: Low confidence, no tools, poor result
5. **Return format**: Always `{"reward": float}`

## 8. Integration with VERL

- Function name: `arc_vision_compute_reward`
- Location: `examples/arc_vision/arc_vision_custom_reward.py`
- Called by: VERL's naive reward manager
- Parameters passed via: `custom_reward_function.reward_kwargs`

## 9. Research Alignment

This implementation enables:
- **Confidence-gated tool learning**: Models learn when tools help
- **Efficient tool use**: Penalties prevent overuse
- **Production readiness**: Rewards align with real-world needs

Without all three components, the model will NOT learn the confidence-aware tool usage behavior that is central to the Arc Vision research contribution.