# Rollout Importance Sampling

**Author:** [Yingru Li](https://richardli.xyz/)

Last updated: 10/27/2025.

This document provides a comprehensive overview of the Rollout Importance Sampling (IS) implementation in verl.

### BibTeX Citation

```bibtex
@misc{liu-li-2025,
  title = {When Speed Kills Stability: Demystifying RL Collapse from the Inference-Training Mismatch},
  url = {https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Inference-Training-Mismatch-271211a558b7808d8b12d403fd15edda},
  author = {Jiacai Liu and Yingru Li and Yuqian Fu and Jiawei Wang and Qian Liu and Yu Shen},
  year = {2025},
  month = september,
}
```

## Overview

Rollout Importance Sampling corrects for distribution mismatch between:
- **Rollout policy**: e.g., vLLM with BFloat16
- **Training policy**: e.g., FSDP with FP32

This mismatch can lead to biased gradient estimates and unstable training. Rollout IS applies importance sampling weights to correct these biases.

### Key Design Principle: Separation of IS Weights and Rejection Sampling

**Important**: As of 10/27/2025, the implementation separates two mechanisms:

1. **IS Weights** (`rollout_is_weights`): Always TRUE ratios π_train/π_rollout
   - Never zeroed, even for rejected samples
   - Preserves true importance ratios for policy gradient calculations

2. **Rejection Sampling** (`modified_response_mask`): Applied via response_mask
   - Mask mode: Excludes tokens/sequences with outlier IS ratios
   - Veto: Excludes sequences with catastrophic tokens
   - Used for loss aggregation (denominator calculation)

This separation ensures:
- ✅ Correct loss normalization (rejected samples excluded from denominator)
- ✅ True IS ratios preserved (not zeroed for rejected samples)
- ✅ Padding positions still zeroed in weights (different from rejection)

## Configuration

```yaml
# Rollout IS configuration (all in algorithm config)
algorithm:
  # Main control: set threshold to enable (null = disabled)
  rollout_is_threshold: 2.0
  # Whether to apply weights to loss (default: false = metrics only)
  rollout_is: true
  rollout_is_threshold_lower: null  # Auto-reciprocal
  rollout_is_level: token
  rollout_is_mode: truncate
  rollout_is_veto_threshold: 1e-4

# REQUIRED: Enable log prob calculation
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

Key features:
- ✅ Three aggregation levels: token, sequence, geometric
- ✅ Two bounding modes: truncate, mask
- ✅ Dual threshold support (upper/lower)
- ✅ Veto mechanism for catastrophic outliers
- ✅ 30+ comprehensive metrics
- ✅ Log-space computation for numerical stability
- ✅ Memory-efficient implementation

## Files

### **Core Implementation**

- `verl/trainer/ppo/mismatch_helper.py` - Contains `compute_rollout_importance_weights()` and `compute_is_metrics()`
- `verl/trainer/ppo/core_algos.py` - Rollout IS integration with PPO
- `verl/workers/actor/dp_actor.py` - Metrics collection and logging

### **Configuration Files**

- `verl/trainer/config/algorithm.py` - Rollout IS parameters in `AlgoConfig`
- `verl/workers/config/actor.py` - Rollout IS parameters in `ActorConfig`
- `verl/trainer/config/actor/actor.yaml` - Rollout IS configuration section
- `verl/trainer/config/ppo_trainer.yaml` - Algorithm config with rollout IS

### **Documentation**

- `docs/examples/config.rst` - Configuration parameter descriptions

### **Example Scripts**

- `recipe/dapo/run_dapo_qwen2.5_32b_rollout_is.sh` - DAPO example with rollout IS
- `examples/rollout_importance_sampling/README.md` - Comprehensive usage guide
- `examples/rollout_importance_sampling/run_with_rollout_is.sh` - Basic example

### **Tests**

- `tests/trainer/ppo/test_rollout_is.py` - Unit tests
- `tests/trainer/ppo/test_rollout_is_integration.py` - Integration tests

## Configuration Parameters

### `algorithm.rollout_is_threshold` (float or null)
**Main on/off switch.** Upper threshold for IS weights.
- `null` = disabled (no computation, no metrics)
- `float` value (e.g., 2.0) = enabled (compute weights and metrics)

### `algorithm.rollout_is` (bool)
Whether to apply IS weights to policy loss. Default: `False`
- `true` = apply weights to loss (full IS correction)
- `false` = compute metrics only (useful for monitoring before enabling)

**Recommended threshold ranges:**
- Token level: 1.5 - 5.0
- Sequence level: 2.0 - 10.0
- Geometric level: 1.0002 - 1.001

### `algorithm.rollout_is_threshold_lower` (float or null)
Lower threshold for IS weights. If `null`, defaults to 1/upper (reciprocal).

### `algorithm.rollout_is_level` (str)
Aggregation level for IS weights:
- `"token"`: Per-token ratios
- `"sequence"`: Product of ratios
- `"geometric"`: Geometric mean (experimental)

### `algorithm.rollout_is_mode` (str)
Bounding mode for handling outlier IS weights:
- `"truncate"`: Clamp weights at upper threshold only (TIS)
  - No lower bound clamping or rejection for outlier ratios
  - IS weights capped at upper threshold to prevent extreme importance ratios
  - **Note**: Veto-based rejection can still occur (see `rollout_is_veto_threshold`)
- `"mask"`: Rejection sampling via response_mask (MIS)
  - Rejects tokens/sequences with IS ratios outside [lower, upper]
  - **Important**: Rejection applied to `response_mask`, NOT by zeroing IS weights
  - IS weights remain as true ratios
  - **Note**: Veto-based rejection also applies (independent mechanism)

### `algorithm.rollout_is_veto_threshold` (float)
Per-token veto threshold for catastrophic outliers.
- If any token has ratio < this threshold, the entire sequence is rejected via `response_mask`
- Default: `1e-4` (detects ratios 10,000x off)
- **Important**: Applied **independently** of `rollout_is_mode` (works in both truncate and mask modes)
- Veto applies rejection to `response_mask`, NOT by zeroing IS weights
- IS weights remain as true ratios even for vetoed sequences

## Usage

### Basic Setup

```yaml
algorithm:
  rollout_is_threshold: 2.0  # Main control
  rollout_is: true           # Apply to loss (default: false)
  rollout_is_level: token
  rollout_is_mode: truncate

actor_rollout_ref:
  rollout:
    calculate_log_probs: true  # Required!
```

### Metrics

All metrics are prefixed with `mismatch/`. For example, `rollout_is_mean` appears as `mismatch/rollout_is_mean` in logs.

#### **Core IS Weight Metrics**

- **`rollout_is_mean`**: Mean importance sampling weight across all valid tokens
  - **Ideal value**: Close to 1.0 (indicates minimal distribution mismatch)
  - **Warning**: < 0.5 or > 2.0 suggests significant policy mismatch

- **`rollout_is_std`**: Standard deviation of IS weights
  - **Ideal value**: < 0.5 for stable training
  - **Warning**: > 1.0 indicates high variance, may need tighter thresholds

- **`rollout_is_min`**: Minimum IS weight observed
  - Shows the most underweighted token/sequence

- **`rollout_is_max`**: Maximum IS weight observed (before truncation/masking)
  - Shows the most overweighted token/sequence
  - Compare with `rollout_is_threshold` to see truncation impact

#### **Effective Sample Size**

- **`rollout_is_eff_sample_size`**: Effective sample size after IS weighting
  - **Formula**: `1 / mean(weights²)` where weights are normalized
  - **Range**: 0.0 to 1.0 (as fraction of original batch)
  - **Ideal value**: > 0.5 (retaining at least 50% effective samples)
  - **Warning**: < 0.3 means high variance, losing too many effective samples

#### **Veto Mechanism Metrics**

- **`rollout_is_veto_fraction`**: Fraction of sequences rejected by veto mechanism
  - **Important**: Sequences are rejected via `response_mask=0`, NOT by zeroing IS weights
  - IS weights remain as true ratios even for vetoed sequences
  - Veto detects catastrophic tokens (ratio < veto_threshold, e.g., < 1e-4)
  - **Ideal value**: < 0.05 (less than 5% vetoed)
  - **Warning**: > 0.1 suggests policies are too different or numerical issues

- **`rollout_is_catastrophic_token_fraction`**: Fraction of tokens below veto threshold
  - Identifies problematic tokens before sequence-level veto is applied
  - Each catastrophic token causes its entire sequence to be rejected
  - **Warning**: > 0.01 indicates widespread distribution issues or numerical instability

#### **Threshold Exceedance Metrics**

- **`rollout_is_ratio_fraction_high`**: Fraction of weights exceeding upper threshold
  - Shows how often truncation/masking occurs on high end
  - **Ideal value**: < 0.1 (most weights within bounds)

- **`rollout_is_ratio_fraction_low`**: Fraction of weights below lower threshold
  - Shows how often masking occurs on low end (mask mode only)
  - **Ideal value**: < 0.1

#### **Sequence-Level Metrics** (for sequence/geometric modes)

- **`rollout_is_seq_mean`**: Mean IS weight at sequence level
  - Should match `rollout_is_mean` for sequence-level aggregation

- **`rollout_is_seq_std`**: Standard deviation of sequence-level IS weights

- **`rollout_is_seq_min`**: Minimum sequence-level IS weight

- **`rollout_is_seq_max`**: Maximum sequence-level IS weight

- **`rollout_is_seq_max_deviation`**: Maximum absolute deviation from 1.0 at sequence level
  - **Ideal value**: < 1.0
  - Shows worst-case sequence mismatch

- **`rollout_is_seq_fraction_high`**: Fraction of sequences exceeding upper threshold

- **`rollout_is_seq_fraction_low`**: Fraction of sequences below lower threshold

#### **Masking Metrics** (mask mode only)

- **`rollout_is_masked_fraction`**: Fraction of tokens rejected via response_mask
  - **Important**: Tokens are rejected by setting `response_mask=0`, NOT by zeroing IS weights
  - IS weights remain as true ratios (π_train/π_rollout)
  - **Ideal value**: < 0.1 (less than 10% rejected)
  - **Warning**: > 0.3 means losing too much data

- **`rollout_is_seq_masked_fraction`**: Fraction of sequences with at least one rejected token
  - Shows sequence-level impact of rejection sampling
  - For token-level: sequence rejected if ANY token is outside [lower, upper]
  - For sequence-level: all tokens have same weight, so entire sequence rejected or accepted

#### **Distribution Mismatch Metrics** (Training vs Rollout Policy)

- **`mismatch_training_ppl`**: Perplexity of training policy (e.g., FSDP FP32)
  - **Formula**: `exp(-mean(log_probs))`
  - Lower is better (model is more confident)

- **`mismatch_rollout_ppl`**: Perplexity of rollout policy (e.g., vLLM BF16)
  - Should be close to `mismatch_training_ppl` if policies match well

- **`mismatch_ppl_ratio`**: Ratio of training PPL to rollout PPL
  - **Formula**: `exp(mean(log(training_ppl / rollout_ppl)))`
  - **Ideal value**: Close to 1.0
  - **Meaning**: > 1.0 means training is less confident than rollout

- **`mismatch_training_log_ppl`**: Log perplexity of training policy
  - Useful for identifying trends (linear scale)

- **`mismatch_rollout_log_ppl`**: Log perplexity of rollout policy

- **`mismatch_log_ppl_diff`**: Mean difference in log perplexities
  - **Formula**: `mean(log_ppl_rollout - log_ppl_training)`
  - **Ideal value**: Close to 0.0
  - Sign indicates which policy is more confident

- **`mismatch_log_ppl_abs_diff`**: Mean absolute log perplexity difference
  - Magnitude of mismatch regardless of direction

- **`mismatch_log_ppl_diff_max`**: Maximum log perplexity difference across sequences
  - Identifies worst-case sequence

- **`mismatch_log_ppl_diff_min`**: Minimum log perplexity difference across sequences

- **`mismatch_kl`**: KL divergence KL(π_rollout || π_training)
  - **Formula**: `mean(log_prob_rollout - log_prob_training)`
  - **Ideal value**: Close to 0.0 (policies match)
  - **Warning**: > 0.1 indicates significant mismatch
  - **Note**: Can be negative (rollout is less confident)

- **`mismatch_k3_kl`**: K3 KL estimator
  - **Formula**: `mean(exp(log_ratio) - log_ratio - 1)`
  - More stable for small KL values
  - Always non-negative

#### **Example: Accessing Metrics in Code**

```python
# Metrics are returned from compute_rollout_importance_weights
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights

# NEW: Returns 3 values (weights, modified_response_mask, metrics)
weights_proto, modified_response_mask, metrics = compute_rollout_importance_weights(
    old_log_prob=training_log_probs,      # from training policy
    rollout_log_prob=rollout_log_probs,   # from rollout policy
    response_mask=response_mask,
    rollout_is_level="token",
    rollout_is_mode="mask",  # Using mask mode for rejection sampling
    rollout_is_threshold=2.0,
    rollout_is_threshold_lower=0.5,
    rollout_is_veto_threshold=1e-4,
)

# Extract IS weights (always true ratios, never zeroed)
is_weights = weights_proto.batch["rollout_is_weights"]

# modified_response_mask has rejection applied
# - Tokens/sequences outside [0.5, 2.0] are masked to 0
# - Sequences with catastrophic tokens are masked to 0
# - IS weights remain as true ratios (NOT zeroed)

# All metrics have 'mismatch/' prefix
print(f"Mean IS weight: {metrics['mismatch/rollout_is_mean']:.3f}")
print(f"Effective sample size: {metrics['mismatch/rollout_is_eff_sample_size']:.3f}")
print(f"Veto fraction: {metrics['mismatch/rollout_is_veto_fraction']:.3f}")
print(f"Masked fraction: {metrics['mismatch/rollout_is_masked_fraction']:.3f}")
print(f"KL divergence: {metrics['mismatch/mismatch_kl']:.3f}")

# Verify IS weights are true ratios (not zeroed)
print(f"\n✓ IS weights min: {is_weights[response_mask.bool()].min():.4f}")
print(f"✓ IS weights max: {is_weights[response_mask.bool()].max():.4f}")
print(f"✓ All IS weights > 0: {(is_weights[response_mask.bool()] > 0).all()}")

# Check rejection via response_mask
rejected_tokens = (response_mask == 1) & (modified_response_mask == 0)
print(f"\n✓ Rejected {rejected_tokens.sum()} tokens via response_mask")
print(f"✓ IS weights for rejected tokens are NON-ZERO (true ratios)")

# Check for warning conditions
if metrics['mismatch/rollout_is_mean'] < 0.5 or metrics['mismatch/rollout_is_mean'] > 2.0:
    print("⚠️  Warning: Mean IS weight far from 1.0, significant policy mismatch detected")

if metrics['mismatch/rollout_is_eff_sample_size'] < 0.3:
    print("⚠️  Warning: Low effective sample size, high variance in IS weights")

if metrics['mismatch/rollout_is_veto_fraction'] > 0.1:
    print("⚠️  Warning: High veto fraction, policies may be too different")
```

#### **Example: Monitoring Metrics During Training**

```python
# In your training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # ... rollout phase ...

        # Compute IS weights and get metrics (NEW: 3 return values)
        weights_proto, modified_response_mask, metrics = compute_rollout_importance_weights(
            old_log_prob=batch.old_log_prob,
            rollout_log_prob=batch.rollout_log_prob,
            response_mask=batch.response_mask,
            rollout_is_level=config.rollout_is_level,
            rollout_is_mode=config.rollout_is_mode,
            rollout_is_threshold=config.rollout_is_threshold,
            rollout_is_threshold_lower=config.rollout_is_threshold_lower,
            rollout_is_veto_threshold=config.rollout_is_veto_threshold,
        )

        # Log to tensorboard/wandb
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step=global_step)

        # IMPORTANT: Update batch response_mask with rejection applied
        batch.response_mask = modified_response_mask

        # Use IS weights in training (true ratios, never zeroed)
        is_weights = weights_proto.batch["rollout_is_weights"]
        # ... apply weights to policy gradient ...
```

#### **Example: Conditional Alerting Based on Metrics**

```python
def check_rollout_is_health(metrics, config):
    """Check if rollout IS metrics indicate healthy training."""
    warnings = []

    # Check mean IS weight
    mean_weight = metrics['mismatch/rollout_is_mean']
    if mean_weight < 0.5 or mean_weight > 2.0:
        warnings.append(f"Mean IS weight {mean_weight:.3f} is far from 1.0")

    # Check effective sample size
    ess = metrics['mismatch/rollout_is_eff_sample_size']
    if ess < 0.3:
        warnings.append(f"Effective sample size {ess:.3f} is too low")

    # Check veto fraction
    veto_frac = metrics['mismatch/rollout_is_veto_fraction']
    if veto_frac > 0.1:
        warnings.append(f"Veto fraction {veto_frac:.3f} is too high")

    # Check variance
    std = metrics['mismatch/rollout_is_std']
    if std > 1.0:
        warnings.append(f"IS weight std {std:.3f} is too high")

    # Check KL divergence
    kl = metrics['mismatch/mismatch_kl']
    if abs(kl) > 0.1:
        warnings.append(f"KL divergence {kl:.3f} indicates significant mismatch")

    if warnings:
        print("⚠️  Rollout IS Health Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        return False
    else:
        print("✅ Rollout IS metrics look healthy")
        return True

# Use in training (NEW: 3 return values)
_, _, metrics = compute_rollout_importance_weights(...)
is_healthy = check_rollout_is_health(metrics, config)

if not is_healthy:
    # Consider adjusting config or investigating issues
    print("Consider:")
    print("  - Tightening rollout_is_threshold")
    print("  - Switching to geometric aggregation level")
    print("  - Checking if rollout and training policies are too different")
```

### Running Examples

Start with the basic token-level truncate configuration:
```bash
bash examples/rollout_importance_sampling/run_with_rollout_is.sh
```

Monitor metrics for 1-2 epochs before adjusting parameters.

## Configuration Examples

### Example 1: Full IS Correction
```yaml
algorithm:
  rollout_is_threshold: 2.0
  rollout_is: true  # Apply weights to loss
  rollout_is_level: token
  rollout_is_mode: truncate
```

### Example 2: Metrics Only (Monitoring Mode)
```yaml
algorithm:
  rollout_is_threshold: 2.0
  rollout_is: false  # Compute metrics, don't apply weights
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
```

### Example 4: Asymmetric Thresholds
```yaml
algorithm:
  rollout_is_threshold: 5.0
  rollout_is: true
  rollout_is_threshold_lower: 0.8
  rollout_is_level: token
  rollout_is_mode: mask
```

## Troubleshooting

### Issue: High variance in IS weights
**Symptoms:** `rollout_is_std` > 1.0, `rollout_is_eff_sample_size` < 0.3

**Solutions:**
1. Switch from `sequence` to `geometric` level
2. Tighten thresholds
3. Verify rollout and training aren't too different

### Issue: Too many sequences vetoed
**Symptoms:** `rollout_is_veto_fraction` > 0.1

**Solutions:**
1. Relax veto threshold: `rollout_is_veto_threshold: 1e-3`
2. Check for numerical issues in log prob computation
3. Verify policies aren't completely different

### Issue: Mean IS weight far from 1.0
**Symptoms:** `rollout_is_mean` < 0.5 or > 2.0

**Solutions:**
1. Verify `calculate_log_probs=True` is set
2. Check rollout_log_probs are correctly passed
3. Check for systematic bias

### Debugging: Visualizing Metrics

**Example: Plot IS weight distribution**

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_is_metrics(metrics_history):
    """Plot rollout IS metrics over training steps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Mean IS weight over time
    axes[0, 0].plot(metrics_history['mismatch/rollout_is_mean'])
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Ideal')
    axes[0, 0].set_title('Mean IS Weight')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].legend()

    # Plot 2: Effective sample size
    axes[0, 1].plot(metrics_history['mismatch/rollout_is_eff_sample_size'])
    axes[0, 1].axhline(y=0.5, color='g', linestyle='--', label='Good')
    axes[0, 1].axhline(y=0.3, color='r', linestyle='--', label='Warning')
    axes[0, 1].set_title('Effective Sample Size')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].legend()

    # Plot 3: Veto fraction
    axes[0, 2].plot(metrics_history['mismatch/rollout_is_veto_fraction'])
    axes[0, 2].axhline(y=0.1, color='r', linestyle='--', label='Warning')
    axes[0, 2].set_title('Veto Fraction')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].legend()

    # Plot 4: KL divergence over time
    axes[1, 0].plot(metrics_history['mismatch/mismatch_kl'], label='KL')
    axes[1, 0].plot(metrics_history['mismatch/mismatch_k3_kl'], label='K3 KL')
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()

    # Plot 5: PPL ratio over time
    axes[1, 1].plot(metrics_history['mismatch/mismatch_ppl_ratio'])
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Ideal')
    axes[1, 1].set_title('PPL Ratio (Training/Rollout)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].legend()

    # Hide unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('rollout_is_metrics.png', dpi=150)
    print("Saved plot to rollout_is_metrics.png")
```

**Example: Metric collection during training**

```python
# Collect metrics over time
metrics_history = {
    'mismatch/rollout_is_mean': [],
    'mismatch/rollout_is_eff_sample_size': [],
    'mismatch/rollout_is_veto_fraction': [],
    'mismatch/mismatch_kl': [],
    'mismatch/mismatch_k3_kl': [],
    'mismatch/mismatch_ppl_ratio': [],
}

# In training loop
for step in range(num_steps):
    # ... compute IS weights ... (NEW: 3 return values)
    _, _, metrics = compute_rollout_importance_weights(...)

    # Store metrics
    for key in metrics_history.keys():
        if key in metrics:
            metrics_history[key].append(metrics[key])

    # Plot every 100 steps
    if step % 100 == 0:
        plot_is_metrics(metrics_history)
```

## Performance Impact

- **Memory overhead**: ~1% of model memory
- **Computational overhead**: 1-3% depending on level
- **Training stability**: Significantly improved when mismatch exists


## Testing

Run the test suite to verify everything works:

```bash
# Basic unit tests
python test_rollout_is.py

# Integration tests (if pytest is available)
pytest tests/trainer/ppo/test_rollout_is_integration.py -v
```

Expected output: All tests pass ✓

## Additional Resources

- **Implementation**: `verl/trainer/ppo/mismatch_helper.py`
- **Examples**: `examples/rollout_importance_sampling/`
- **DAPO Example**: `recipe/dapo/run_dapo_qwen2.5_32b_rollout_is.sh`

## Summary

Rollout Importance Sampling provides:
- ✅ Robust handling of distribution mismatch
- ✅ Numerical stability
- ✅ Comprehensive metrics for monitoring
- ✅ Flexibility for different scenarios
- ✅ Memory-efficient computation

## References

- [When Speed Kills Stability: Demystifying RL Collapse from the Inference-Training Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Inference-Training-Mismatch-271211a558b7808d8b12d403fd15edda)
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)