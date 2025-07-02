# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl is a sophisticated Reinforcement Learning library for Large Language Models (LLMs) that implements the HybridFlow framework for efficient RLHF (Reinforcement Learning from Human Feedback) training.

## Key Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run CPU-only unit tests
python -m pytest tests/unit tests/e2e -m "not gpu"

# Run GPU tests
python -m pytest tests/unit tests/e2e -m "gpu"

# Run specific test file
python -m pytest tests/unit/test_specific.py

# Run with coverage
python -m pytest --cov=verl tests/
```

### Linting and Type Checking
```bash
# Format code with yapf
yapf -i -r verl/

# Run linter
ruff check verl/

# Type checking (if available)
mypy verl/
```

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[all]"

# Build documentation
cd docs && make html
```

### Running Examples
```bash
# PPO training example
cd examples/ppo_trainer
python3 train_ppo.py data.train_data_path=/path/to/data \
    model.model_path=/path/to/model \
    data.output_dir=ppo_gsm8k

# RLHF data collection
cd examples/rlhf_data_collection
python3 data_gen.py
```

## Architecture Overview

### Core Components

1. **verl/trainer/** - High-level training orchestration
   - `main_generation.py` - Text generation during training
   - `main_trainer.py` - Main PPO training loop
   - `main_reward.py` - Reward computation
   - `fsdp_*` - Distributed training implementations

2. **verl/models/** - Model implementations
   - Supports HuggingFace, vLLM, SGLang backends
   - Multi-modal support for vision-language models

3. **verl/rl/** - RL algorithm implementations
   - PPO, PPO2, GRPO, DAPO algorithms
   - Core PPO components in `verl/trainer/ppo/`

4. **verl/workers/** - Distributed computing workers
   - Actor, Critic, Reference policy, Reward model workers
   - Ray-based distribution in `verl/trainer/ppo/ray_*`

5. **verl/utils/** - Utility functions
   - Configuration management using Hydra
   - Tokenization, padding, data processing utilities

### Key Design Patterns

- **HybridFlow**: Efficient data flow between generation and training phases
- **Worker Pattern**: Separate workers for different model roles (actor, critic, etc.)
- **Modular Backends**: Pluggable inference engines (vLLM, SGLang)
- **Config-Driven**: Hydra configs for all major components

## Adding New Features

### Adding a New Model
1. Create model class in `verl/models/`
2. Register in appropriate registry
3. Add configuration in `verl/trainer/config/`
4. Add tests in `tests/unit/`

### Adding a New RL Algorithm
1. Implement algorithm class in `verl/rl/`
2. Create corresponding rollout/trainer modules
3. Add example in `examples/`
4. Document in `docs/algorithms/`

### Performance Optimization
- Use FSDP for distributed training
- Enable gradient checkpointing for memory efficiency
- Configure batch sizes based on GPU memory
- Use mixed precision training when appropriate

## Code Style Guidelines

- Line length limit: 120 characters (enforced by ruff)
- No emojis in code or documentation
- Professional tone for enterprise open-source
- Follow existing patterns in neighboring files
- Use type hints where beneficial

## Code Quality Principles

Write production-ready code that prioritizes clarity, performance, and maintainability. Follow these principles:

- Favor explicit over implicit behavior
- Write self-documenting code with clear variable/function names
- Minimize dependencies and complexity
- Include error handling and logging for production debugging
- Optimize for readability by future developers (including yourself)
- Test critical paths and edge cases
- Follow existing codebase patterns and conventions

Default to simple, proven solutions over clever optimizations. Every line should serve a clear purpose.

## Testing Best Practices

- Write unit tests for new utilities
- Use pytest markers for GPU/CPU test separation
- Mock external dependencies appropriately
- Test both single-GPU and multi-GPU scenarios

## Common Issues and Solutions

### Memory Issues
- Reduce batch size or sequence length
- Enable gradient checkpointing
- Use FSDP with appropriate sharding strategy

### Distributed Training
- Ensure Ray is properly initialized
- Check NCCL environment variables
- Verify GPU visibility with CUDA_VISIBLE_DEVICES

### Model Loading
- Verify model path and format compatibility
- Check tokenizer configuration matches model
- Ensure correct backend selection (HF vs vLLM)

## Important Configuration Files

- `pyproject.toml` - Project dependencies and build configuration
- `.github/workflows/` - CI/CD pipeline definitions
- `examples/*/config/` - Hydra configuration examples
- `docs/conf.py` - Sphinx documentation configuration