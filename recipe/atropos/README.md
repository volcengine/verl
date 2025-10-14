# Atropos-VeRL Integration

This directory contains the integration between VeRL and Atropos RL environments, enabling training with environment feedback and token-level advantage overrides.

## Overview

The Atropos integration provides:

- **GRPO Algorithm** with token-level advantage overrides from environments
- **Environment Feedback** from Atropos API for computing advantages
- **Service Orchestration** for managing Atropos environments and VeRL training
- **Error Handling** with retry logic and fallback mechanisms

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
- `AtroposEnvironmentClient`: Handles API communication with Atropos (exponential-backoff retry, health-check)
- `AtroposGRPOComputer`: Broadcasts env-level advantages to tokens via `response_mask`, falls back to standard GRPO when needed

### 2. `grpo_atropos_trainer.py`
GRPO trainer implementation:
- Extends `RayPPOTrainer` while keeping the standard GRPO estimator
- Injects token-level advantages from Atropos when provided
- Falls back to standard GRPO automatically when overrides are unavailable

### 3. `launch_atropos_verl_services.py`
Service orchestration script:
- Automatically launches Atropos environment server
- Starts vLLM inference workers
- **Registers** the vLLM endpoint back to Atropos (if the env supports `/register_inference_endpoint`)
- Manages service lifecycle, health checks and cleanup

### 4. `core_algos.py` (modified)
- Extended the GRPO estimator to accept optional token-level advantage overrides

## Usage

### Quick Start (Automated)

Use the automated launcher to start all components:

```bash
# Launch all services automatically
python recipe/atropos/launch_atropos_verl_services.py \
    --config recipe/atropos/config/gsm8k_grpo_example.yaml
```

For a CLI-only workflow that mirrors the standard VeRL examples, use the provided script:

```bash
# Override any flags by appending them to the command
recipe/atropos/run_qwen2_5-3b_atropos_grpo.sh trainer.atropos.api_url=http://localhost:9001
```

This will:
1. Start the Atropos environment server
2. Launch vLLM inference workers (if configured)
3. Begin the GRPO training process

### Manual Setup

#### Prerequisites

1. Start Atropos environment server:
```bash
cd /path/to/atropos
python environments/gsm8k_server.py serve --slurm false
```

2. Verify Atropos is running:
```bash
curl http://localhost:9001/health
```

#### Training with GRPO-Atropos

##### Using Example Script
```bash
cd verl/recipe/atropos
python example_gsm8k_grpo.py --config-name gsm8k_grpo_example
```

##### Using Configuration File
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
  adv_estimator: grpo  # Use standard GRPO estimator
  
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
- Exponential-backoff retry logic for API calls
- Fallback to standard GRPO if Atropos unavailable
- Comprehensive error logging & early shape-assertions
- Graceful degradation (training continues on fallback)

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

## Scaling to Multi-Node

For multi-node training, configure your environment:

```bash
# Set Ray cluster head
export MASTER_ADDR=<head_node_ip>
export MASTER_PORT=29500

# Launch with Ray cluster
ray start --head --port=10001  # On head node
ray start --address=<head_node_ip>:10001  # On worker nodes

# Update config for distributed training
trainer:
  ray:
    runtime_env:
      working_dir: "."
    resources:
      head_node_ip: <head_node_ip>
```

For SLURM environments, see `examples/slurm/ray_on_slurm.slurm`.

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure Atropos server is running or increase `retry_attempts` / `retry_delay`
2. **Timeout Errors**: Increase `timeout` in configuration
3. **Shape Mismatch**: Ensure `response_mask` length matches advantage length returned by Atropos
4. **Retry Limit Hit**: Check cumulative wait time vs `max_wait_time` setting

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
