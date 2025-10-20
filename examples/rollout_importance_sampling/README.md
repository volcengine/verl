# Rollout Importance Sampling (IS) Examples

This directory contains examples and documentation for using Rollout Importance Sampling to correct distribution mismatch between rollout and training policies.

**References:**
- When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
- Off-policy RL: https://fengyao.notion.site/off-policy-rl

## Overview

Rollout Importance Sampling corrects for distribution mismatch when:
1. **Rollout generation** uses one policy (e.g., vLLM with BFloat16)
2. **Training** uses another policy (e.g., FSDP with FP32)
3. This mismatch leads to biased gradient estimates

## Quick Start

### Basic Configuration

```yaml
algorithm:
  # Main control: set threshold to enable (null = disabled)
  rollout_is_threshold: 2.0
  # Whether to apply weights to policy loss (true) or just compute metrics (false)
  rollout_is: true
  rollout_is_level: token
  rollout_is_mode: truncate

# IMPORTANT: Must enable log prob calculation
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### Running the Example

```bash
# Basic example with token-level truncate
bash examples/rollout_importance_sampling/run_with_rollout_is.sh
```

## Configuration Options

### Aggregation Levels (`rollout_is_level`)

| Level | Properties | Threshold Range |
|-------|-----------|-----------------|
| **token** | Per-token | 1.5 - 5.0 |
| **sequence** | Per-sequence | 2.0 - 10.0 |
| **geometric** | Geometric mean | 1.0002 - 1.001 |

### Bounding Modes (`rollout_is_mode`)

| Mode | Behavior |
|------|----------|
| **truncate** | Cap weights at upper threshold only |
| **clip** | Zero out weights outside [lower, upper] |

### Key Parameters

- `rollout_is_threshold`: Upper threshold for IS weights (null = disabled, float = enabled). **Main on/off switch.**
- `rollout_is`: Whether to apply weights to loss (true) or just compute metrics (false). Default: false.
- `rollout_is_threshold_lower`: Lower threshold (null = auto 1/upper)
- `rollout_is_veto_threshold`: Catastrophic outlier threshold (default: 1e-4)

## Configuration Examples

### Example 1: Full IS Correction (Apply Weights)

```yaml
algorithm:
  rollout_is_threshold: 2.0
  rollout_is: true  # Apply to loss
  rollout_is_level: token
  rollout_is_mode: truncate
  rollout_is_veto_threshold: 1e-4
```

### Example 2: Metrics Only (No Weight Application)

```yaml
algorithm:
  rollout_is_threshold: 2.0
  rollout_is: false  # Compute metrics only, don't apply to loss
  rollout_is_level: token
  rollout_is_mode: truncate
```

### Example 3: Geometric Mean with Mask

```yaml
algorithm:
  rollout_is_threshold: 1.0002
  rollout_is: true
  rollout_is_threshold_lower: 0.9998
  rollout_is_level: geometric
  rollout_is_mode: mask
  rollout_is_veto_threshold: 1e-4
```

### Example 4: Sequence-level with Truncate

```yaml
algorithm:
  rollout_is_threshold: 5.0
  rollout_is: true
  rollout_is_threshold_lower: null  # Auto-reciprocal: 0.2
  rollout_is_level: sequence
  rollout_is_mode: truncate
  rollout_is_veto_threshold: 1e-4
```

### Example 5: Asymmetric Thresholds

```yaml
algorithm:
  rollout_is_threshold: 5.0
  rollout_is: true
  rollout_is_threshold_lower: 0.8
  rollout_is_level: token
  rollout_is_mode: mask
```

## Monitoring Metrics

Key metrics to watch (all prefixed with `mismatch/` in logs):

### Health Indicators
- `rollout_is_mean`: Mean IS weight across sequences
- `rollout_is_eff_sample_size`: Effective sample size after weighting
- `rollout_is_veto_fraction`: Fraction of sequences vetoed

### Distribution Metrics
- `rollout_is_max`, `rollout_is_min`: Weight extremes
- `rollout_is_std`: Standard deviation

### Diagnostic Metrics
- `rollout_is_ratio_fraction_high`: Fraction exceeding upper threshold
- `rollout_is_ratio_fraction_low`: Fraction below lower threshold
- `rollout_is_catastrophic_token_fraction`: Catastrophic tokens detected

### Mismatch Metrics (Training vs Rollout Policy)

These metrics help diagnose the distribution mismatch between rollout and training policies:

**Perplexity Metrics:**
- `mismatch_training_ppl`: Perplexity of training policy
- `mismatch_rollout_ppl`: Perplexity of rollout policy
- `mismatch_ppl_ratio`: Ratio of training PPL to rollout PPL
- `mismatch_log_ppl_diff`: Log perplexity difference

**KL Divergence Metrics:**
- `mismatch_kl`: KL divergence KL(π_rollout || π_training)
- `mismatch_k3_kl`: K3 KL estimator

## Troubleshooting

### Issue: High Variance in IS Weights

**Symptoms**: `rollout_is_std` > 1.0, `rollout_is_eff_sample_size` < 0.3

**Solutions**:
1. Switch from `sequence` to `geometric` level
2. Tighten thresholds
3. Check if rollout and training are too different

### Issue: Too Many Sequences Vetoed

**Symptoms**: `rollout_is_veto_fraction` > 0.1

**Solutions**:
1. Relax veto threshold: `rollout_is_veto_threshold: 1e-3`
2. Check for numerical issues in log prob computation
3. Verify rollout and training policies aren't completely different

### Issue: Mean IS Weight Far from 1.0

**Symptoms**: `rollout_is_mean` < 0.5 or > 2.0

**Solutions**:
1. Check that `calculate_log_probs=True` is set
2. Verify rollout_log_probs are correctly passed
3. Check for systematic bias in rollout vs training

### Issue: Too Much Data Discarded (Mask Mode)

**Symptoms**: `rollout_is_masked_fraction` > 0.5

**Solutions**:
1. Widen thresholds
2. Switch to `truncate` mode
3. Use `geometric` level for better stability

## Performance Considerations

### Memory Usage
- Rollout IS adds minimal memory overhead (~1% of model memory)
- Log-space computation prevents numerical overflow

### Computational Cost
- Token-level: ~1-2% overhead
- Sequence-level: ~2-3% overhead
- Geometric: ~2-3% overhead

## Advanced Topics

### Dual Thresholds

Specify both upper and lower explicitly:

```yaml
rollout_is_threshold: 2.0      # Upper
rollout_is_threshold_lower: 0.5  # Lower (not 1/2.0 = 0.5)
```

Or use auto-reciprocal:

```yaml
rollout_is_threshold: 2.0      # Upper = 2.0, Lower = 0.5 (auto)
rollout_is_threshold_lower: null
```

### Veto Mechanism

The veto mechanism zeros out entire sequences containing catastrophic outliers:

- If any token has ratio < `rollout_is_veto_threshold`, the entire sequence is rejected
- This prevents extreme outliers from dominating training
- Default threshold: 1e-4 (ratio 10,000x off)
- Set to `null` to disable: `rollout_is_veto_threshold: null`

## Examples

See the script in this directory:
- `run_with_rollout_is.sh`: Basic example with token-level truncate mode

## References

- Implementation: `verl/trainer/ppo/mismatch_helper.py`
- Core algorithm: `verl/trainer/ppo/core_algos.py`
- Paper: "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"
