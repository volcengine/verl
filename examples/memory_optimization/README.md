# Memory Optimization Examples

This directory contains examples demonstrating memory optimization techniques in verl for training large language models with limited GPU memory.

## Overview

Training large models often requires careful memory management. verl provides several memory optimization features that trade compute for reduced memory usage, enabling training of larger models or using larger batch sizes.

## Memory Optimization Features

### 1. Activation Offloading

Offloads activations to CPU during the forward pass and reloads them during the backward pass. This significantly reduces GPU memory usage at the cost of additional CPU-GPU data transfer.

```yaml
actor_rollout_ref:
  model:
    enable_activation_offload: True
    enable_gradient_checkpointing: True  # Usually combined
```

**Note**: Activation offloading is only available for FSDP/FSDP2 backends.

### 2. Gradient Checkpointing

Also known as activation checkpointing. Instead of storing activations for the backward pass, recomputes them during backpropagation. Reduces memory at the cost of ~33% additional compute.

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
```

### 3. Parameter and Optimizer Offloading

Offloads model parameters and optimizer states to CPU memory. Useful for very large models that don't fit in GPU memory.

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      param_offload: True
      optimizer_offload: True
```

### 4. reshard_after_forward (Full Sharding)

Enables ZeRO-3 style full parameter sharding. Parameters are gathered for computation and resharded immediately after, minimizing memory footprint.

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      reshard_after_forward: True  # Default is True
```

### 5. Dynamic Batch Sizing

Instead of fixed batch sizes, dynamically packs sequences to maximize token throughput within a memory budget.

```yaml
actor_rollout_ref:
  actor:
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 8192  # Maximum tokens per GPU
```

### 6. Inference Memory Configuration

Control how much GPU memory the inference engine (vLLM/SGLang) uses, leaving room for training.

```yaml
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.4  # Lower when training on same GPU
    enforce_eager: True          # Disable CUDA graphs to save memory
```

### 7. CPU Memory Optimization

For CPU OOM issues caused by memory fragmentation, use efficient memory allocators:

```bash
# Using tcmalloc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# Using jemalloc
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

## Example Scripts

### `run_memory_optimized_ppo.sh`

A comprehensive example demonstrating all memory optimization features for PPO training. This script is configured for memory-constrained environments.

**Usage:**
```bash
# Basic usage
./run_memory_optimized_ppo.sh

# With custom model
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct ./run_memory_optimized_ppo.sh

# With additional overrides
./run_memory_optimized_ppo.sh trainer.n_gpus_per_node=4
```

## Configuration Guidelines

### Memory-Constrained (Single GPU or Limited VRAM)

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
    enable_activation_offload: True
  actor:
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 4096  # Reduce if OOM
    fsdp_config:
      param_offload: True
      optimizer_offload: True
  rollout:
    gpu_memory_utilization: 0.3
    enforce_eager: True
```

### Multi-GPU with Large Models (70B+)

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
    enable_activation_offload: True
  actor:
    fsdp_config:
      param_offload: True
      optimizer_offload: True
      reshard_after_forward: True
  ref:
    fsdp_config:
      param_offload: True  # Reference model definitely offload
  rollout:
    gpu_memory_utilization: 0.8
    tensor_model_parallel_size: 4
```

### Balanced Performance/Memory

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
    enable_activation_offload: False  # Skip for better speed
  actor:
    use_dynamic_bsz: True
    fsdp_config:
      param_offload: False
      optimizer_offload: False
      reshard_after_forward: True
  rollout:
    gpu_memory_utilization: 0.5
```

## Troubleshooting

### GPU OOM During Training

1. Enable gradient checkpointing: `enable_gradient_checkpointing=True`
2. Enable activation offloading: `enable_activation_offload=True`
3. Reduce `ppo_max_token_len_per_gpu`
4. Enable parameter offloading: `param_offload=True`

### GPU OOM During Rollout

1. Reduce `gpu_memory_utilization` (e.g., 0.3-0.5)
2. Disable CUDA graphs: `enforce_eager=True`
3. Reduce `max_num_seqs` or `max_num_batched_tokens`

### CPU OOM

1. Use tcmalloc or jemalloc via `LD_PRELOAD`
2. Reduce offloading if CPU memory is limited
3. Check for memory leaks in data loading

## References

- [Issue #144: Additional memory optimization features](https://github.com/volcengine/verl/issues/144)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [TorchTune Activation Offloading](https://github.com/pytorch/torchtune)
