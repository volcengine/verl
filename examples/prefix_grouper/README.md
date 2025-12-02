# PrefixGrouper Examples

This directory contains examples for using **PrefixGrouper**, an optimization technique that groups samples by shared prompts to reduce redundant computations in GRPO.

## Introduction

> Official Repository: [https://github.com/johncaged/PrefixGrouper](https://github.com/johncaged/PrefixGrouper)

``PrefixGrouper`` is a plug-and-play efficient GRPO training tool that requires minimal modifications to existing codebases to achieve reduced computation, lower device memory consumption, and accelerated training.

In current mainstream GRPO training pipelines, policy model training primarily involves copying prefixes (typically questions, multimodal inputs, etc.) `G` times. Consequently, when training data prefixes are sufficiently long (e.g., long-context reasoning, image/long-video inference), redundant computation during training becomes non-negligible.

**PrefixGrouper** decomposes the original redundant self-attention operation into prefix self-attention + suffix concat-attention.

<h3 align="center">
    <img src="https://raw.githubusercontent.com/johncaged/PrefixGrouper/main/assets/images/method.jpg">
</h3>

## Installation

```bash
pip install prefix_grouper
```

## Limitations

- Currently only supports FSDP worker (Megatron worker is not supported yet).
- Incompatible with `balance_batch=True`.
- Incompatible with `use_dynamic_bsz=True`.
- Incompatible with `use_remove_padding=True` (Flash Attention V2 variable length).
- Incompatible with `use_fused_kernels=True`.
- Incompatible with Ulysses sequence parallelism (`use_ulysses_sp=True`) and ring-attention.

## Directory Structure

- `qwen3/modeling_qwen3.py`: A modified Qwen3 model implementation that supports `PrefixGrouper`.
- `run_qwen3_pg.sh`: An example training script enabling PrefixGrouper.

## How to Use

### 1. Prepare the Model

PrefixGrouper requires the model to accept a `prefix_grouper` argument in its `forward` method and use it to compute attention.

The provided `qwen3/modeling_qwen3.py` demonstrates how to integrate PrefixGrouper into the Qwen model. Key changes include:
- Importing `PrefixGrouper`.
- Adding `prefix_grouper` argument to `forward`.
- Passing `prefix_grouper` to the flash attention mechanism.

**Option A (Recommended): Use `trust_remote_code=True`**
1. Copy `qwen3/modeling_qwen3.py` to your local model checkpoint directory (e.g., `/path/to/your/model/`).
2. Ensure your model's `config.json` has the correct `auto_map` to load this file.
   ```json
   "auto_map": {
       "AutoModelForCausalLM": "modeling_qwen3.Qwen3ForCausalLM"
   }
   ```
3. In the training script, ensure `trust_remote_code=True` is set (default in many verl scripts).

**Option B: Monkey Patching**
You can also create a custom python entry script that imports this model class and registers it to `AutoModel` before training starts.

### 2. Run Training

Use the provided script `run_qwen3_pg.sh` to start training. This script is pre-configured with:
- `actor_rollout_ref.actor.use_prefix_grouper=True`
- `trainer.balance_batch=False`
- `actor_rollout_ref.model.use_remove_padding=False`

```bash
bash examples/prefix_grouper_examples/run_qwen3_pg.sh
```

## Performance

**Benchmark Results** (Qwen3-4B, 4×H800, `rollout.n=4`):

| Context Length | Metric | PG | No PG | Speedup |
|----------------|--------|-----|-------|---------|
| **4K** | `old_log_prob` | 1.31s | 1.70s | **1.30x** |
| | `update_actor` | 4.80s | 6.07s | **1.26x** |
| | `step` | 17.08s | 19.40s | **1.14x** |
| **8K** | `old_log_prob` | 1.69s | 2.63s | **1.56x** |
| | `update_actor` | 5.98s | 10.18s | **1.70x** |
| | `step` | 19.48s | 24.71s | **1.27x** |

As context length increases, the speedup becomes more pronounced.
