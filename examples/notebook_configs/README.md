# Example Notebook Configurations

This directory contains example configurations for common use cases with the verl Jupyter notebooks.

## Available Examples

### 1. `grpo_gsm8k_sglang.py`
- **Algorithm**: GRPO
- **Dataset**: GSM8K (math problems)
- **Backend**: SGLang
- **Hardware**: 8x A100-80GB (single node)
- **Best for**: Fast training on math tasks

### 2. `ppo_math_vllm.py`
- **Algorithm**: PPO (with critic)
- **Dataset**: GSM8K + MATH (merged)
- **Backend**: vLLM
- **Hardware**: 8x A100-80GB (single node)
- **Best for**: Multi-dataset training with stable PPO

### 3. `grpo_single_gpu.py`
- **Algorithm**: GRPO
- **Dataset**: GSM8K
- **Backend**: vLLM
- **Hardware**: 1x RTX 4090 (24GB)
- **Best for**: Testing on single GPU, memory-efficient training

## How to Use These Examples

These files are **reference examples** showing complete configurations. You **don't need to run them directly**. Instead:

### Method 1: Copy Values into Notebook

Open `1_verl_complete_training.ipynb` and copy the relevant config dictionaries:

```python
# Example: Using grpo_gsm8k_sglang.py values

# In Section 1.5:
BACKEND = 'sglang'

# In Section 2:
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 8,
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
}

# In Section 3:
DATA_CONFIG = {
    'train_files': '/home/user/data/gsm8k/train.parquet',
    'val_files': '/home/user/data/gsm8k/test.parquet',
    'max_prompt_length': 512,
    'max_response_length': 1024,
}

MODEL_CONFIG = {
    'model_path': 'Qwen/Qwen3-8B',
    'output_dir': './checkpoints/grpo_gsm8k_sglang',
}

# In Section 4 (GRPO):
# The GRPO_SPECIFIC settings are already in the notebook,
# just verify they match the example
```

### Method 2: Load as Python Module

You can also import these files directly in the notebook:

```python
# In a notebook cell:
import sys
sys.path.append('/home/user/verl/examples/notebook_configs')

import grpo_gsm8k_sglang as example_config

# Then use the values:
BACKEND = example_config.BACKEND
DATA_CONFIG = example_config.DATA_CONFIG
MODEL_CONFIG = example_config.MODEL_CONFIG
# etc.
```

## Customizing Examples

Feel free to modify these examples for your specific needs:

### Change the Model

```python
# Instead of:
MODEL_CONFIG = {
    'model_path': 'Qwen/Qwen3-8B',
    # ...
}

# Use your own:
MODEL_CONFIG = {
    'model_path': 'meta-llama/Llama-3.1-8B',
    # ...
}
```

### Change the Dataset

```python
# Instead of GSM8K:
DATA_CONFIG = {
    'train_files': '/home/user/data/custom/train.parquet',
    'val_files': '/home/user/data/custom/test.parquet',
    # ...
}
```

### Adjust for Your Hardware

If you have fewer GPUs:

```python
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 4,  # Changed from 8
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,  # Adjusted TP
}
```

## Creating Your Own Examples

To create a custom configuration:

1. Copy one of the existing example files
2. Rename it (e.g., `my_custom_config.py`)
3. Modify the values for your use case
4. Reference it in your notebook

## Tips

### Memory Optimization

If you're running out of memory:
- Enable offloading: `param_offload=True`, `optimizer_offload=True`
- Reduce batch sizes: Lower `ppo_micro_batch_size_per_gpu`
- Use gradient checkpointing: `enable_gradient_checkpointing=True`
- Reduce sample count: Lower `actor_rollout_ref.rollout.n`

### Speed Optimization

If training is too slow:
- Disable offloading: `param_offload=False`, `optimizer_offload=False`
- Increase batch sizes (if memory allows)
- Use SGLang backend (generally faster)
- Increase tensor parallel size

### Stability

If training is unstable (NaN, diverging):
- Lower learning rate: Try `1e-7` instead of `1e-6`
- Increase KL coefficient: `kl_loss_coef=0.01`
- Use PPO instead of GRPO (more stable)
- Check your reward function

## Questions?

See the main [notebooks/README.md](../../notebooks/README.md) for more detailed documentation and troubleshooting.
