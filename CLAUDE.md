# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl** (Volcano Engine Reinforcement Learning for LLMs) is an open-source RL training library for large language models, implementing the HybridFlow paper (EuroSys 2025). It uses a hybrid-controller programming model that separates control flow (RL algorithm logic) from computation flow (training/inference engines).

## Development Commands

### Installation

```bash
# Install with vLLM backend (most common)
pip install -e .[test,vllm]

# Install with SGLang backend
pip install -e .[test,sglang]

# Basic installation
pip install -e .[test]
```

### Code Quality (Pre-commit)

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always mypy
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

### Testing

```bash
# CPU tests (tests ending in _on_cpu.py)
pytest tests/**/test_*_on_cpu.py

# GPU tests (default)
pytest tests/

# Specific test directories
pytest tests/trainer/
pytest tests/models/
pytest tests/workers/

# Quick sanity checks
pytest tests/special_sanity/
```

**Test categories:**
- `test_*_on_cpu.py` - CPU-only tests
- `special_distributed/` - Multi-GPU unit tests
- `special_e2e/` - End-to-end training tests
- `special_npu/` - NPU-specific tests
- `special_sanity/` - Quick sanity tests
- `special_standalone/` - Standalone environment tests

### Documentation

```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html
python -m http.server -d _build/html/
```

### Auto-generate Config Files

After modifying trainer configs, regenerate via:
```bash
bash scripts/generate_trainer_config.sh
```

This updates `_generated_ppo_trainer.yaml` and `_generated_ppo_megatron_trainer.yaml`.

## Core Architecture

### Data Transfer: DataProto

`verl/protocol.py` - **DataProto** is the unified data transfer protocol for ALL data movement between functions/modules. Built on TensorDict, it standardizes data passing throughout the RL dataflow.

Key operations:
- `batch`: TensorDict containing tensors with same batch size
- `non_tensor_batch`: dict of numpy arrays for non-tensor data (strings, objects)
- `meta_info`: dict for metadata (config, metrics, etc.)

```python
# Create from dict
data = DataProto.from_dict(tensors={...}, non_tensors={...})

# Access and manipulate
sliced = data[10:20]  # Returns DataProto
selected = data[[1, 5, 10]]  # Returns DataProto
item = data[5]  # Returns DataProtoItem (single item)

# Split/concat
chunks = data.chunk(4)  # List[DataProto]
merged = DataProto.concat([data1, data2])
```

### Single-Controller Pattern

`verl/single_controller/` - Implements the single-process controller that orchestrates distributed workers via Ray.

**Key decorators:**
- `@register` - Register worker functions
- `@Dispatch` - Dispatch calls to workers (RPC-like)
- `@Execute` - Execute on specific worker

### Trainer Architecture

`verl/trainer/` - Main training orchestration.

**Entry points:**
- `main_ppo.py` - PPO training entry
- `main_generation.py` - Generation/testing
- `sft_trainer.py` - Supervised fine-tuning

**PPO components:**
- `ppo/ray_trainer.py` - Ray-based trainer
- `ppo/core_algos.py` - Algorithm implementations (PPO, GRPO, RLOO, etc.)

### Workers (Distributed Execution)

`verl/workers/` - Distributed workers for different backend integrations.

**Backend workers:**
- `fsdp_workers.py` - PyTorch FSDP backend
- `megatron_workers.py` - Megatron-LM backend

**Specialized workers:**
- `actor/` - Actor model training
- `critic/` - Critic model workers
- `rollout/` - Generation (vLLM, SGLang, HF)
- `reward_manager/` - Reward computation
- `reward_model/` - Reward model workers

### Checkpoint Engine

`verl/checkpoint_engine/` - Unified weight synchronization between training and inference backends.

- `naive` - torch.distributed all_gather (default, highest elasticity)
- `nccl` - NCCL-based (high performance, fixed clusters)
- `nixl` - NIXL-based (high elasticity, heterogeneous hardware)

### Configuration System

Uses Hydra for configuration management. Configs in `verl/trainer/config/*.yaml`.

**To enable FSDP2** (recommended):
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
reward_model.strategy=fsdp2
```

## Supported Algorithms

Core algorithms in `verl/config/algorithm.py`:
- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization)
- **RLOO** (Reinforcement Learning from Offline Outcomes)
- **REINFORCE++** variants
- **ReMax**
- And more: GSPO, DAPO, SPPO, DrGRPO, OTB, OPPO, etc.

## Model Support

`verl/models/` - Model loading and support.

- `transformers/` - HuggingFace Transformers integration
- `mcore/` - Megatron core models (DeepSeek, Qwen3, etc.)
- Specific architectures: `qwen2/`, `llama/`, etc.

Supported: Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, VLMs (Qwen2.5-vl, Kimi-VL)

## Key Patterns and Conventions

### PR Title Format

```
[{module}] {type}: {description}
```

Modules: `fsdp`, `megatron`, `vllm`, `sglang`, `rollout`, `trainer`, `ci`, etc.
Types: `feat`, `fix`, `refactor`, `chore`, `test`

### Recipe Submodule

The `recipe/` directory is a git submodule for external RL recipes:
- Initialize: `git submodule update --init --recursive recipe`
- URL: https://github.com/verl-project/verl-recipe.git

Experimental features kept in `verl/experimental/`:
- `fully_async_policy` - Fully async PPO
- `one_step_off_policy` - Off-policy training
- `vla` - Vision-Language Action models
- `agent_loop` - Agent training loop

## Hardware Support

- NVIDIA GPUs (CUDA) - Primary support
- AMD GPUs (ROCm)
- Ascend NPUs
- Mixed hardware via NIXL checkpoint engine

## Important Files

- `verl/protocol.py` - DataProto implementation
- `verl/base_config.py` - BaseConfig dataclass
- `verl/trainer/config/` - Hydra configs
- `scripts/generate_trainer_config.sh` - Config auto-generation
