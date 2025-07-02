# Data Format Fixes Summary

## Issues Fixed

### 1. JSON Serialization Error (AttributeError)
**Error**: `AttributeError: 'str' object has no attribute 'get'`
**Location**: `verl/utils/dataset/rl_dataset.py`, line 307
**Cause**: `extra_info` and `reward_model` were stored as JSON strings instead of dictionaries
**Fix**: Removed `json.dumps()` calls and store as native Python dictionaries

### 2. Image Format (Already Correct)
**Format**: `"images": [{"image": image_path}]`
**Why**: Aligns with Qwen2.5-VL's expected format for image inputs

## Complete Data Format

The correct VERL-compatible data format for multi-modal training:

```python
{
    "data_source": "rootsautomation/ScreenSpot",
    "prompt": [  # List of message dicts, not JSON string
        {
            "role": "user",
            "content": "Enhanced prompt with reasoning..."
        }
    ],
    "images": [{"image": "/path/to/image.png"}],  # Dict format for Qwen2.5-VL
    "ability": "ui_detection",
    "ground_truth": [0.1, 0.2, 0.3, 0.4],  # Normalized bbox for reward
    "reward_model": {  # Dict, not JSON string
        "style": "arc_vision",
        "ground_truth": [0.1, 0.2, 0.3, 0.4],
        "confidence_threshold": 0.7,
        "reward_weights": {
            "task": 0.6,
            "tool": 0.3,
            "gate": 0.1
        }
    },
    "extra_info": {  # Dict, not JSON string
        "split": "train",
        "index": 0,
        "original_instruction": "Click on...",
        "original_bbox": [100, 200, 300, 400],
        "element_type": "button",
        "screenshot_id": "train_0"
    }
}
```

## Key Lessons

1. **VERL expects native Python types** in parquet files:
   - Lists should be Python lists
   - Dicts should be Python dicts
   - Don't use `json.dumps()` unless specifically needed

2. **Image format for Qwen2.5-VL**:
   - Use `[{"image": path}]` format
   - The processor will handle loading via `fetch_image()`

3. **Follow existing examples**:
   - Check `examples/data_preprocess/gsm8k.py` for text-only format
   - Check `examples/data_preprocess/geo3k.py` for multi-modal format

## Regeneration Steps

```bash
# Remove old data
rm -rf /root/data/arc_vision/screenspot/*.parquet

# Regenerate with fixed script
cd /root/verl/examples/arc_vision
python prepare_screenspot_data.py --split_test_data --max_samples 1200

# Verify files
ls -la /root/data/arc_vision/screenspot/
```

The training should now proceed without data format errors.