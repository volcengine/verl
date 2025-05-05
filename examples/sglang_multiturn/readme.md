# Multi-Turn Rollout Example (GSM8K)

This example demonstrates how to perform **multi-turn rollout** using SGLang with a tool-calling capable model (e.g., Qwen2.5-3B) on the GSM8K dataset.

## Usage


### Step 1: Download GSM8K Dataset

```bash
cd examples/data_preprocess
python3 gsm8k.py
```

This will download and preprocess the GSM8K dataset into ~/data/gsm8k/.

### Step 2: Run Multi-Turn Rollout
If you have 8 GPUs
Use the standard 8-GPU script:

```
cd your_verl_root_dir
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

If you have only 4 GPUs
Use the fallback 4-GPU script:

```
cd your_verl_root_dir
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh 
```

# Notes

- The rollout supports multi-turn conversations with tool-calling capabilities.

- Current tools are used for GSM8K answer evaluation.

- Future versions may extend to search and code interpreter tools.