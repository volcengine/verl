# Atropos-VeRL GRPO Integration

This directory contains the complete implementation of Atropos-VeRL integration, enabling GRPO training with multi-environment coordination through the Atropos API.

## 🎯 Overview

The Atropos-VeRL integration provides:

- **Full Atropos API Integration**: Complete implementation of `/register`, `/endpoints`, `/scored_data`, and `/batch` endpoints
- **Token-level Advantage Support**: Native support for token-level advantages from Atropos environments
- **Automatic Weight Synchronization**: Policy weights are automatically synchronized between training and inference
- **Production-Ready Infrastructure**: FSDP distributed training, Ray orchestration, robust error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VeRL Trainer  │    │  Atropos API    │    │ Atropos Envs    │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ GRPO Train  │◄┼────┼►│ /register   │ │    │ │Environment 1│ │
│ │             │ │    │ │ /batch      │ │    │ │Environment 2│ │
│ └─────────────┘ │    │ │ /scored_data│ │    │ │Environment N│ │
│                 │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │                 │    │                 │
│ │Inference    │◄┼────┼─────────────────┼────┼─────────────────┤
│ │Servers      │ │    │                 │    │                 │
│ │(vLLM/SGLang)│ │    │                 │    │                 │
│ └─────────────┘ │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Atropos API Server**: Ensure your Atropos server is running and accessible
2. **VeRL Environment**: Set up VeRL with required dependencies
3. **GPU Resources**: CUDA-compatible GPUs for training and inference

### Basic Usage

```bash
# Set environment variables
export ATROPOS_API_URL="http://localhost:8000"
export MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
export OUTPUT_DIR="/tmp/verl_atropos_checkpoints"

# Run the integration
cd verl/examples/atropos_trainer
bash run_atropos_grpo.sh
```

### Advanced Configuration

```bash
# Custom configuration
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf" \
LORA_RANK=16 \
BATCH_SIZE=512 \
NUM_GPUS=4 \
INFERENCE_BACKEND="sglang" \
bash run_atropos_grpo.sh
```

## 📋 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATROPOS_API_URL` | `http://localhost:8000` | Atropos API endpoint |
| `MODEL_PATH` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model path |
| `OUTPUT_DIR` | `/tmp/verl_atropos_checkpoints` | Checkpoint directory |
| `NUM_GPUS` | `8` | Number of GPUs to use |
| `BATCH_SIZE` | `1024` | Training batch size |
| `MAX_EPOCHS` | `100` | Maximum training epochs |
| `LORA_RANK` | `0` | LoRA rank (0 = full fine-tuning) |
| `INFERENCE_BACKEND` | `vllm` | Inference backend (`vllm` or `sglang`) |

### Configuration File

The integration uses `atropos_grpo_trainer.yaml` for detailed configuration:

```yaml
# Atropos-specific settings
atropos:
  api_url: "http://localhost:8000"
  timeout: 30
  batch_retry_attempts: 8
  batch_retry_delay: 0.3
  batch_max_wait_time: 12.0

# Algorithm configuration
algorithm:
  adv_estimator: grpo_atropos  # Use Atropos-aware GRPO
  use_kl_in_reward: false

# GRPO settings
actor_rollout_ref:
  rollout:
    name: "atropos"  # Use Atropos rollout worker
    n: 5  # Number of samples per prompt
```

## 🔧 Implementation Details

### Core Components

#### 1. AtroposRolloutWorker (`verl/workers/atropos_workers.py`)

The main worker that coordinates with Atropos:

```python
class AtroposRolloutWorker(ActorRolloutRefWorker):
    def generate_sequences(self, prompts, sampling_params=None, **kwargs):
        # 1. Register with Atropos API
        self._register_with_atropos()
        
        # 2. Generate responses using VeRL inference
        generated_data = super().generate_sequences(prompts, sampling_params, **kwargs)
        
        # 3. Submit scored data to Atropos
        self._submit_scored_data_to_atropos(tokens, masks, scores)
        
        # 4. Retrieve processed batch with advantages
        atropos_batch = self._retrieve_batch_from_atropos()
        
        # 5. Convert to VeRL format
        return self._convert_atropos_to_verl_data(atropos_batch)
```

#### 2. GRPO_ATROPOS Advantage Estimator (`verl/trainer/ppo/core_algos.py`)

Extended GRPO algorithm supporting token-level advantages:

```python
@register_adv_est(AdvantageEstimator.GRPO_ATROPOS)
def compute_grpo_atropos_advantage(token_level_rewards, response_mask, index, **kwargs):
    # Detect token-level vs response-level advantages
    has_token_level_advantages = torch.any(torch.std(token_level_rewards, dim=-1) > epsilon)
    
    if has_token_level_advantages:
        # Use token-level advantages with group normalization
        return process_token_level_advantages(token_level_rewards, response_mask, index)
    else:
        # Fall back to standard GRPO
        return compute_standard_grpo(token_level_rewards, response_mask, index)
```

#### 3. Atropos Utilities (`verl/utils/atropos_utils.py`)

Helper functions for API integration:

- `AtroposAPIValidator`: API connectivity and data validation
- `AtroposDataConverter`: Data format conversion between VeRL and Atropos
- `AtroposEndpointManager`: Inference server endpoint management
- `AtroposRetryHandler`: Robust retry logic with exponential backoff

### API Integration

#### Registration Flow

```python
# 1. Register trainer with Atropos
registration_data = {
    "wandb_group": "verl_atropos_integration",
    "wandb_project": "verl_atropos_grpo", 
    "batch_size": 1024,
    "max_token_len": 1536,
    "checkpoint_dir": "/tmp/verl_checkpoints",
    "starting_step": 0,
    "num_steps": 100
}

response = requests.post(f"{atropos_url}/register", json=registration_data)
trainer_uuid = response.json()['uuid']
```

#### Data Submission

```python
# 2. Submit scored data to Atropos
submission_data = {
    "trainer_uuid": trainer_uuid,
    "step": current_step,
    "data": {
        "tokens": tokens.tolist(),
        "masks": masks.tolist(),
        "scores": scores.tolist(),
        "ref_logprobs": ref_logprobs.tolist()  # Optional
    }
}

requests.post(f"{atropos_url}/scored_data", json=submission_data)
```

#### Batch Retrieval

```python
# 3. Retrieve processed batch with advantages
response = requests.get(f"{atropos_url}/batch")
batch_data = response.json()

# Each item in batch contains:
# - tokens: List[int] - Token IDs
# - masks: List[float] - Attention masks  
# - advantages: List[float] or List[List[float]] - Token or response level
# - scores: List[float] - Environment scores
```

## 🔍 Monitoring and Debugging

### Logging

The integration provides comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Key log messages to monitor:
# - "Atropos API connectivity confirmed"
# - "Trainer successfully registered with UUID: ..."
# - "Successfully submitted scored data for step X"
# - "Successfully retrieved batch with X items"
```

### Common Issues

#### 1. API Connectivity

```bash
# Test Atropos API connectivity
curl -s --fail "$ATROPOS_API_URL/status"
```

**Solution**: Ensure Atropos server is running and accessible at the configured URL.

#### 2. Batch Retrieval Timeout

```
WARNING: Maximum wait time (12.0s) exceeded
WARNING: Failed to retrieve batch from Atropos after all retry attempts
```

**Solution**: 
- Increase `batch_max_wait_time` in configuration
- Check Atropos environment processing speed
- Verify sufficient environments are running

#### 3. Data Format Errors

```
ERROR: Invalid registration data format
ERROR: Missing required registration field: batch_size
```

**Solution**: Verify configuration matches Atropos API requirements.

### Performance Monitoring

Monitor key metrics:

- **API Response Times**: Track `/batch` endpoint latency
- **Batch Processing Rate**: Batches processed per minute
- **Training Throughput**: Tokens processed per second
- **Memory Usage**: GPU memory utilization during inference

## 🧪 Testing

### Unit Tests

```bash
# Run Atropos integration tests
cd verl
python -m pytest tests/test_atropos_integration.py -v
```

### Integration Testing

```bash
# Test with mock Atropos server
python examples/atropos_trainer/test_integration.py
```

### End-to-End Testing

```bash
# Full integration test with real Atropos server
ATROPOS_API_URL="http://your-atropos-server:8000" \
python examples/atropos_trainer/run_atropos_grpo.sh
```

## 📊 Performance Considerations

### Optimization Tips

1. **Batch Size Tuning**: Larger batches improve GPU utilization but increase memory usage
2. **Inference Backend**: SGLang may provide better throughput for certain models
3. **LoRA Training**: Use LoRA (rank 16-64) for faster training with large models
4. **Parallel Environments**: Run multiple Atropos environments for higher throughput

### Scaling Guidelines

| Model Size | Recommended GPUs | Batch Size | LoRA Rank |
|------------|------------------|------------|-----------|
| 3B | 2-4 | 512-1024 | 0-32 |
| 7B | 4-8 | 256-512 | 16-64 |
| 13B | 8-16 | 128-256 | 32-128 |
| 70B+ | 16+ | 64-128 | 64-256 |

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .

# Install development dependencies
pip install pytest black isort mypy
```

### Code Style

```bash
# Format code
black verl/workers/atropos_workers.py
isort verl/workers/atropos_workers.py

# Type checking
mypy verl/workers/atropos_workers.py
```

### Adding New Features

1. **Extend AtroposRolloutWorker**: Add new API endpoints or functionality
2. **Update Configuration**: Add new settings to `atropos_grpo_trainer.yaml`
3. **Add Tests**: Include unit and integration tests
4. **Update Documentation**: Update this README and code comments

## 📚 References

- [VeRL Documentation](https://github.com/volcengine/verl)
- [Atropos Framework](https://github.com/NousResearch/atropos)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://sgl-project.github.io/)

## 📄 License

This integration is licensed under the Apache License 2.0. See the LICENSE file for details. 