# Fix for VERL Data Format Issues

## Error
```
AttributeError: 'str' object has no attribute 'get'
File "/root/verl/verl/utils/dataset/rl_dataset.py", line 307, in __getitem__
    index = row_dict.get("extra_info", {}).get("index", 0)
```

## Root Cause
VERL's `rl_dataset.py` expects `extra_info` and `reward_model` fields to be dictionaries, but we were storing them as JSON strings using `json.dumps()`. When VERL tries to call `.get()` on these fields, it fails because they are strings, not dictionaries.

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

The fix changes how dictionaries are stored in the parquet files:

**Before (incorrect)**:
```python
"reward_model": json.dumps({
    "style": "arc_vision",
    "ground_truth": bbox_normalized,
    ...
}),
"extra_info": json.dumps({
    "split": split,
    "index": idx,
    ...
})
```

**After (correct)**:
```python
"reward_model": {
    "style": "arc_vision", 
    "ground_truth": bbox_normalized,
    ...
},
"extra_info": {
    "split": split,
    "index": idx,
    ...
}
```

Key changes:
- Removed `json.dumps()` calls for `reward_model` and `extra_info`
- Store these fields as native Python dictionaries
- This matches the format used in other VERL examples (gsm8k.py, geo3k.py)

## Expected Behavior

After regenerating the data, the training should proceed past the data loading step without the AttributeError.