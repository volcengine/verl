# Rollout Correction Examples

This directory contains examples and documentation for using Rollout Correction to address off-policy issues in RL training.

**References:**
- When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
- Off-policy RL: https://fengyao.notion.site/off-policy-rl

## Overview

Rollout Correction addresses off-policy issues including:
1. **Policy mismatch**: Rollout policy (e.g., vLLM BFloat16) vs Training policy (e.g., FSDP FP32)
2. **Model staleness**: Training on trajectories from older policy checkpoints
3. **Distribution shifts**: Any distribution gap between data collection and training

Rollout Correction uses importance sampling (IS) weights and rejection sampling (RS) to correct for these distribution shifts.

## Quick Start

### Basic Configuration

```yaml
algorithm:
  rollout_correction:
    rollout_is: token  # IS weights: token/sequence/null
    rollout_is_threshold: 2.0  # Upper threshold
    rollout_rs: null  # Rejection sampling: token/sequence/geometric/null
    rollout_rs_threshold: null
    rollout_rs_threshold_lower: null
    rollout_token_veto_threshold: null  # Veto threshold

# IMPORTANT: Must enable log prob calculation
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

### Running the Example

```bash
# Basic example with token-level truncate
bash examples/rollout_correction/run_with_rollout_corr.sh
```

## Configuration Options

### IS Weights Aggregation Levels (`rollout_is`)

| Level | Properties | Threshold Range |
|-------|-----------|-----------------|
| **token** | Per-token weighting | 1.5 - 5.0 |
| **sequence** | Per-sequence weighting | 2.0 - 10.0 |
| **null** | Disabled | N/A |

### Rejection Sampling Modes (`rollout_rs`)

| Mode | Behavior | Threshold Range |
|------|----------|-----------------|
| **token** | Per-token rejection | 1.5 - 5.0 |
| **sequence** | Per-sequence rejection | 2.0 - 10.0 |
| **geometric** | Geometric mean rejection | 1.0002 - 1.001 |
| **null** | Disabled | N/A |

### Key Parameters

- `rollout_is`: IS weights aggregation level (`token`, `sequence`, or `null`)
- `rollout_is_threshold`: Upper threshold for IS weights
- `rollout_rs`: Rejection sampling mode (`token`, `sequence`, `geometric`, or `null`)
- `rollout_rs_threshold`: RS upper threshold
- `rollout_rs_threshold_lower`: RS lower threshold (null = auto 1/upper)
- `rollout_token_veto_threshold`: Per-token catastrophic outlier veto threshold (null = disabled)

## Configuration Examples

### Example 1: IS Weights Only (Token-level)

```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: null  # No rejection sampling
```

### Example 2: Rejection Sampling Only

```yaml
algorithm:
  rollout_correction:
    rollout_is: null  # No IS weights
    rollout_rs: sequence
    rollout_rs_threshold: 3.0
```

### Example 3: Combined IS + RS

```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: token
    rollout_rs_threshold: 2.0
```

### Example 4: Geometric Mean RS with Veto

```yaml
algorithm:
  rollout_correction:
    rollout_is: null
    rollout_rs: geometric
    rollout_rs_threshold: 1.0002
    rollout_rs_threshold_lower: 0.9998
    rollout_token_veto_threshold: 1e-4
```

### Example 5: Full Configuration

```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: token
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5
    rollout_token_veto_threshold: 1e-4
```

## Monitoring Metrics

Key metrics to watch (all prefixed with `rollout_corr/` in logs):

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
- `training_ppl`: Perplexity of training policy
- `rollout_ppl`: Perplexity of rollout policy
- `ppl_ratio`: Ratio of training PPL to rollout PPL
- `log_ppl_diff`: Log perplexity difference

**KL Divergence Metrics:**
- `kl`: KL divergence KL(π_rollout || π_training)
- `k3_kl`: K3 KL estimator

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
1. Relax veto threshold in config:
   ```yaml
   algorithm:
     rollout_correction:
       rollout_token_veto_threshold: 1e-3
   ```
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
- Rollout Correction adds minimal memory overhead (~1% of model memory)
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

- If any token has ratio < `rollout_token_veto_threshold`, the entire sequence is rejected
- This prevents extreme outliers from dominating training
- Default: `null` (disabled by default)
- Set to `1e-4` to enable (catches ratios 10,000x off)

## Examples

See the script in this directory:
- `run_with_rollout_corr.sh`: Basic example with token-level truncate mode

## References

- Implementation: `verl/trainer/ppo/rollout_corr_helper.py`
- Core algorithm: `verl/trainer/ppo/core_algos.py`
- Paper: "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"
