# Fix for Data Format Error

## Error
```
TypeError: string indices must be integers, not 'str'
```

## Root Cause
The ScreenSpot data preparation script was storing messages as JSON-serialized strings, but VERL's RLHFDataset expects the prompt field to contain a list of message dictionaries directly.

## Solution

The data preparation script has been fixed. You need to **regenerate your data files** with the updated script.

### Steps to Fix

1. **Pull the latest code**:
   ```bash
   git pull origin main
   ```

2. **Remove old data files**:
   ```bash
   rm -rf /root/data/arc_vision/screenspot/*.parquet
   ```

3. **Regenerate the data with the fixed script**:
   ```bash
   cd /root/verl/examples/arc_vision
   python prepare_screenspot_data.py --split_test_data --max_samples 1200
   ```

4. **Verify the new files**:
   ```bash
   ls -la /root/data/arc_vision/screenspot/
   # Should show: train.parquet, validation.parquet, test.parquet
   ```

5. **Run training again**:
   ```bash
   bash run_arc_vision_3b.sh
   ```

## What Changed

The fix changes how messages are stored in the parquet files:

**Before (incorrect)**:
```python
"prompt": json.dumps(messages)  # JSON string: '[{"role": "user", "content": "..."}]'
```

**After (correct)**:
```python
"prompt": messages  # Python list: [{"role": "user", "content": "..."}]
```

Additionally:
- Added `ground_truth` field at the top level for the reward function
- Kept other fields (reward_model, extra_info) as JSON strings as expected

## Expected Behavior

After regenerating the data, the training should proceed past the filtering step without the TypeError.