# Atropos-VeRL Integration Recipe

This recipe provides a production-ready integration between [Atropos](https://github.com/NousResearch/atropos) RL environments and VeRL, enabling training with real environment feedback.

## Overview

The Atropos integration enables:
- **Real Environment Feedback**: Train models using actual Atropos environments (GSM8K, HumanEval, etc.)
- **Token-Level Advantages**: Support for fine-grained advantage computation at the token level
- **Automatic Weight Synchronization**: Seamless policy updates between training and inference
- **Multi-Environment Support**: Train on multiple Atropos environments simultaneously

## Key Components

### 1. Atropos API Client (`atropos_api_client.py`)
A robust client for interacting with the Atropos API server:
- Trainer registration and environment discovery
- Rollout data submission to environments
- Batch retrieval with retry logic
- Advantage extraction and processing

### 2. Ray Trainer (`atropos_ray_trainer.py`)
Extended Ray-based PPO trainer with Atropos integration:
- Real-time environment feedback for advantages
- Fallback to standard GRPO when needed
- Atropos-specific metrics tracking
- Seamless integration with VeRL's distributed infrastructure

### 3. Custom Advantage Estimator
Registered as `grpo_atropos` in `verl/trainer/ppo/core_algos.py`:
- Supports token-level advantage overrides from environments
- Compatible with standard GRPO normalization
- Handles both response-level and token-level advantages

## Quick Start

### Prerequisites

1. **Install VeRL** (if not already installed):
```bash
pip install -e .
```

2. **Install and Start Atropos**:
```bash
# Clone Atropos
git clone https://github.com/NousResearch/atropos.git
cd atropos
pip install -e .

# Start Atropos API server
python -m atroposlib.api --port 9001

# In another terminal, start GSM8K environment
python environments/gsm8k_environment.py
```

### Training with GSM8K

Run the example training script:

```bash
cd verl
bash recipe/atropos/run_atropos_gsm8k.sh
```

Or with custom configuration:

```bash
python -m recipe.atropos.main_atropos_gsm8k \
    --config recipe/atropos/config/atropos_gsm8k.yaml \
    --atropos-url http://localhost:9001
```

## Configuration

### Atropos Configuration

In your training config, add:

```yaml
trainer:
  atropos:
    api_url: http://localhost:9001  # Atropos API server URL
    timeout: 30                      # API timeout in seconds
    use_advantages: true             # Use environment advantages
    fallback_to_grpo: true          # Fallback to GRPO if needed
    retry_attempts: 10              # Batch retrieval retries
    retry_delay: 0.5                # Initial retry delay
    max_wait_time: 30.0             # Maximum wait for batch

algorithm:
  adv_estimator: grpo_atropos       # Use Atropos advantage estimator
```

### Model Configuration

The integration supports any model compatible with VeRL:

```yaml
model:
  partial_pretrain: Qwen/Qwen2.5-Math-1.5B-Instruct
  # or
  # partial_pretrain: meta-llama/Llama-3.2-1B-Instruct
  # partial_pretrain: google/gemma-2b-it
```

## Advanced Usage

### Custom Environment Integration

To use other Atropos environments:

1. Start the desired environment:
```bash
# For code generation
python environments/humaneval_environment.py

# For instruction following
python environments/instruction_following_environment.py
```

2. Update your training data to match the environment's expected format

3. Adjust generation parameters as needed:
```yaml
actor_rollout_ref:
  rollout:
    temperature: 0.2  # Lower for code generation
    max_new_tokens: 512  # Higher for complex tasks
```

### Multi-Environment Training

Train on multiple environments simultaneously:

```python
# In your main script
atropos_config = AtroposConfig(
    api_url="http://localhost:9001",
    # ... other config
)

# Atropos will automatically distribute rollouts 
# across all registered environments based on weights
```

### Monitoring

The integration provides detailed metrics:
- `atropos/mean_score`: Average environment scores
- `atropos/{env_name}/mean_advantage`: Per-environment advantages
- `atropos/num_environments`: Active environment count

## Implementation Details

### Rollout Flow

1. **Generation**: VeRL generates responses using current policy
2. **Submission**: Responses sent to Atropos with metadata
3. **Environment Processing**: Atropos environments evaluate responses
4. **Advantage Computation**: Environments compute token-level advantages
5. **Training**: VeRL updates policy using environment feedback

### Weight Synchronization

The integration uses VeRL's sharding manager pattern:

```python
with self.sharding_manager:
    # Inference engine automatically receives latest weights
    responses = self.inference_engine.generate(prompts)
```

### Error Handling

- Automatic retry with exponential backoff for API calls
- Graceful fallback to standard GRPO when Atropos unavailable
- Comprehensive error messages with troubleshooting steps

## Troubleshooting

### "Cannot connect to Atropos API"

1. Ensure Atropos server is running:
```bash
python -m atroposlib.api --port 9001
```

2. Check the API URL in your config matches the server

3. Verify network connectivity:
```bash
curl http://localhost:9001/status
```

### "No batch received from Atropos"

1. Check that environments are registered and running
2. Verify data format matches environment expectations
3. Increase `max_wait_time` in config if needed

### Low Training Rewards

1. Ensure environment is providing meaningful feedback
2. Check that advantages are being computed correctly
3. Adjust generation parameters (temperature, top_p, etc.)

## Example Results

Training on GSM8K with Qwen2.5-Math-1.5B-Instruct:
- Initial accuracy: ~40%
- After 1000 steps: ~55%
- Convergence: ~65-70%

Note: Results vary based on model size, hyperparameters, and environment configuration.

## Contributing

When extending this integration:

1. Keep all code in the `recipe/atropos` directory
2. Use the established API client for all Atropos communication
3. Ensure compatibility with VeRL's distributed training
4. Add tests for new functionality
5. Document environment-specific requirements

## References

- [Atropos Repository](https://github.com/NousResearch/atropos)
- [VeRL Documentation](https://github.com/volcengine/verl)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)