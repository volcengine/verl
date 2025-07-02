# Validation Generation in FSDP SFT Trainer

This document explains how to use the new validation generation feature in the FSDP SFT Trainer.

## Overview

The FSDP SFT Trainer now supports generating samples during validation, allowing you to:
- Monitor the quality of model outputs during training
- Log generated samples to tracking systems (WandB, SwanLab, MLflow)
- Print sample outputs to console for quick inspection

## Configuration

### Environment Variables (Recommended)

Set environment variables to enable validation generation:

```bash
export VERL_GENERATION_TEMPERATURE=1.0
export VERL_GENERATION_TOP_K=50
export VERL_GENERATION_TOP_P=0.7
export VERL_GENERATION_DO_SAMPLE=true
export VERL_GENERATION_MAX_NEW_TOKENS=100
export VERL_MAX_VAL_SAMPLES=5
```

### Configuration File (Alternative)

Add generation parameters to your configuration file:

```yaml
# Generation parameters for validation
generation:
  temperature: 1.0
  top_k: 50
  top_p: 0.7
  do_sample: true
  max_new_tokens: 100

# Trainer configuration
trainer:
  # Number of validation samples to generate (default: 5)
  max_val_samples: 5
```

### Generation Parameters

- `temperature`: Controls randomness in generation (default: 1.0)
- `top_k`: Number of highest probability tokens to consider (default: 50)
- `top_p`: Cumulative probability threshold for nucleus sampling (default: 0.7)
- `do_sample`: Whether to use sampling instead of greedy decoding (default: true)
- `max_new_tokens`: Maximum number of tokens to generate (default: 100)

### Trainer Parameters

- `max_val_samples`: Maximum number of validation samples to generate (default: 5)

## Usage

### Running with Generation

Use environment variables (recommended):

```bash
# Set generation parameters
export VERL_GENERATION_TEMPERATURE=1.0
export VERL_GENERATION_TOP_K=50
export VERL_GENERATION_TOP_P=0.7
export VERL_GENERATION_DO_SAMPLE=true
export VERL_GENERATION_MAX_NEW_TOKENS=100
export VERL_MAX_VAL_SAMPLES=5

# Run training
python -m verl.trainer.fsdp_sft_trainer
```

Or use the example configuration:

```bash
python -m verl.trainer.fsdp_sft_trainer \
    --config-path config \
    --config-name sft_trainer_with_generation
```

### Custom Configuration

Create your own configuration file extending the base trainer:

```yaml
defaults:
  - sft_trainer
  - _self_

generation:
  temperature: 0.8
  top_k: 40
  top_p: 0.9
  do_sample: true
  max_new_tokens: 150

trainer:
  max_val_samples: 10
  project_name: "my_sft_project"
  experiment_name: "with_validation_generation"
```

## Output

### Console Output

During validation, you'll see output like:

```
=== Validation Samples (Step 1000) ===
Prompt 1: What is the capital of France?
Generation 1: The capital of France is Paris. Paris is located in the north-central part of the country...
--------------------------------------------------
Prompt 2: Explain quantum computing in simple terms.
Generation 2: Quantum computing is a type of computing that uses quantum mechanical phenomena...
--------------------------------------------------
```

### Tracking System Logs

Samples are also logged to your configured tracking system (WandB, SwanLab, etc.) with metrics like:
- `val/sample_1_prompt`
- `val/sample_1_generation`
- `val/sample_2_prompt`
- `val/sample_2_generation`
- etc.

## Implementation Details

### Generation Method

The `generate_samples` method:
1. Takes a batch of input data
2. Applies generation parameters from config
3. Uses the model's `generate` method with proper configuration
4. Decodes the generated sequences
5. Returns both prompts and generations as text

### Validation Integration

Generation is integrated into both:
- Regular epoch validation
- Early exit validation (when `total_training_steps` is reached)

### Error Handling

Generation errors are caught and logged as warnings, allowing training to continue even if generation fails.

## Limitations

- Generation is limited to the first few validation batches to avoid performance impact
- Generated text is truncated in logs to avoid overwhelming output
- Generation parameters are global (not per-sample)
- Only works with causal language models that support the `generate` method

## Troubleshooting

### Common Issues

1. **Generation fails**: Check that your model supports the `generate` method
2. **Memory issues**: Reduce `max_new_tokens` or `max_val_samples`
3. **Poor quality samples**: Adjust `temperature`, `top_k`, or `top_p` parameters

### Debugging

Enable debug logging to see more details:

```bash
export VERL_SFT_LOGGING_LEVEL=DEBUG
```

## Future Enhancements

Potential improvements:
- Per-sample generation parameters
- More sophisticated sampling strategies
- Integration with evaluation metrics
- Support for different generation backends 