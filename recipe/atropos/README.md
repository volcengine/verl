# Atropos-VeRL Integration

This directory contains the integration between VeRL and Atropos RL environments, enabling training with environment feedback and token-level advantage overrides.

## Overview

The Atropos integration provides:

- **GRPO Algorithm** with token-level advantage overrides from environments
- **Environment Feedback** from Atropos API for computing advantages
- **Service Orchestration** for managing Atropos environments and VeRL training
- **Production-Ready Code** with proper error handling and fallback mechanisms

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  VeRL GRPO Trainer  │         │   Atropos API        │
│  ┌──────────────┐   │         │  ┌────────────────┐ │
│  │ GRPO Atropos │   │────────▶│  │ GSM8K Env      │ │
│  │ Computer     │   │ API     │  │ Evaluation     │ │
│  └──────────────┘   │ Calls   │  └────────────────┘ │
│  ┌──────────────┐   │         │  ┌────────────────┐ │
│  │ Ray Workers  │   │         │  │ Advantage      │ │
│  │ (FSDP)       │   │◀────────│  │ Computation    │ │
│  └──────────────┘   │ Token   │  └────────────────┘ │
└─────────────────────┘ Advs    └──────────────────────┘
```

## Key Components

### 1. `atropos_integration.py`
Core integration module with:
- `AtroposEnvironmentClient`: Handles API communication with Atropos
- `AtroposGRPOComputer`: Computes advantages with environment overrides

### 2. `grpo_atropos_trainer.py`
GRPO trainer implementation:
- Extends `RayPPOTrainer` for GRPO algorithm
- Integrates token-level advantages from Atropos
- Provides fallback to standard GRPO

### 3. `core_algos.py` (modified)
- Added `grpo_atropos` advantage estimator
- Supports token-level advantage overrides

## Usage

### Prerequisites

1. Start Atropos environment server:
```bash
cd /path/to/atropos
python environments/gsm8k_server.py serve --slurm false
```

2. Verify Atropos is running:
```bash
curl http://localhost:9001/health
```

### Training with GRPO-Atropos

#### Using Example Script
```bash
cd verl/recipe/atropos
python example_gsm8k_grpo.py --config-name gsm8k_grpo_example
```

#### Using Configuration File
```bash
cd verl
python -m verl.trainer.main_ppo \
    --config-path recipe/atropos/config \
    --config-name gsm8k_grpo_example \
    trainer_cls=recipe.atropos.grpo_atropos_trainer.RayGRPOAtroposTrainer
```

### Configuration

Key configuration options in `config/gsm8k_grpo_example.yaml`:

```yaml
algorithm:
  adv_estimator: grpo_atropos  # Use Atropos GRPO estimator
  
trainer:
  atropos:
    api_url: http://localhost:9001  # Atropos API endpoint
    use_advantages: true            # Use environment advantages
    fallback_to_grpo: true         # Fallback if API unavailable
```

## API Integration

The integration communicates with Atropos via REST API:

### Endpoint: `/evaluate_and_compute_advantages`

**Request:**
```json
{
  "prompts": ["list of prompts"],
  "responses": ["list of responses"],
  "metadata": {}
}
```

**Response:**
```json
{
  "advantages": [/* token-level advantages */],
  "metrics": {
    "mean_reward": 0.85,
    "accuracy": 0.9
  }
}
```

## Metrics and Monitoring

The integration tracks:
- Token-level advantages from environments
- Environment-specific metrics (e.g., accuracy)
- Fallback usage statistics
- API response times

## Error Handling

The implementation includes:
- Automatic retry logic for API calls
- Fallback to standard GRPO if Atropos unavailable
- Comprehensive error logging
- Graceful degradation

## Development

### Adding New Environments

1. Implement environment in Atropos
2. Ensure it provides token-level advantages
3. Update configuration with environment-specific settings

### Testing

Run integration tests:
```bash
pytest recipe/atropos/tests/
```

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure Atropos server is running
2. **Timeout Errors**: Increase `timeout` in configuration
3. **Shape Mismatch**: Check tokenizer compatibility

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger("verl.recipe.atropos").setLevel(logging.DEBUG)
```

## References

- [Atropos Documentation](https://github.com/nousresearch/atropos)
- [VeRL Documentation](https://github.com/volcengine/verl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)