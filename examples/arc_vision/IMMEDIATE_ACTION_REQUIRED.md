# IMMEDIATE ACTION REQUIRED

## The Problem
You're still using OLD data files that have JSON strings instead of dictionaries. The code has been fixed, but you need to regenerate your data.

## The Solution (2 Steps)

### Step 1: Delete Old Data
```bash
rm -rf ~/data/arc_vision/screenspot/*.parquet
rm -rf ~/data/arc_vision/screenspot/images/
```

### Step 2: Regenerate Data with Fixed Format
```bash
cd /root/verl/examples/arc_vision
python3 prepare_screenspot_data.py \
    --local_dir ~/data/arc_vision/screenspot \
    --max_samples 1200 \
    --split_test_data
```

### Step 3: Run Training
```bash
bash run_arc_vision_3b_fixed.sh
```

## Why This Fixes It

The old data has:
```python
"extra_info": '{"index": 0, "split": "train", ...}'  # JSON string
```

The new data has:
```python
"extra_info": {"index": 0, "split": "train", ...}  # Python dict
```

VERL expects dictionaries, not JSON strings. The code is already fixed - you just need fresh data!

## Verification
After regenerating, you can verify with:
```python
import pandas as pd
df = pd.read_parquet('~/data/arc_vision/screenspot/train.parquet')
print(type(df.iloc[0]['extra_info']))  # Should be dict, not str
```