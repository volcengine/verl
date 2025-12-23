# Rollout Correction

**Author:** [Yingru Li](https://richardli.xyz/)

Last updated: 10/30/2025.

---

> **📖 Documentation Structure**
> - **This document** - Practical usage guide: configurations, presets, troubleshooting
> - **[Mathematical Formulations](rollout_corr_math.md)** - Theoretical foundations, derivations, and algorithmic details
>
> Start here for implementation, refer to the math doc for theory and design rationale.

---

This document provides a comprehensive overview of the Rollout Correction implementation in verl.

**Note on Naming**: This feature is called "Rollout Correction" to reflect the complete functionality: importance sampling (IS) weights, rejection sampling (RS), and veto mechanism. The internal variable `rollout_is_weights` retains its name as it specifically refers to the IS weights component.

### BibTeX Citation

```bibtex
@online{liu-li-2025-rl-collapse,
  title = {When Speed Kills Stability: Demystifying {RL} Collapse from the Training-Inference Mismatch},
  author = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei and Liu, Qian and Shen, Yu},
  year = {2025},
  month = sep,
  url = {https://richardli.xyz/rl-collapse}
}
```

### Blog Series

- Main blog post: https://richardli.xyz/rl-collapse
- [Part 1: Why Mismatch Breaks LLM-RL](https://richardli.xyz/rl-collapse-1) (analytical framework using TV distance for bias and χ²-divergence for variance)
- [Part 2: The Gradient Estimator Trials](https://richardli.xyz/rl-collapse-2) (token-level vs sequence-level correction bias-variance tradeoff)
- [Part 3: When Math Meets Reality—Toxic Tails and Length Traps](https://richardli.xyz/rl-collapse-3) (why rejection over clipping, and geometric-level RS)

## Overview

Rollout Correction provides a unified framework to handle **general off-policy problems** in RL training. Any scenario where the data collection distribution differs from the training distribution can benefit from these methods.

**Common off-policy scenarios:**

1. **Policy Mismatch** (Implementation Differences)
   - Different precision: FP8 vs FP16 vs BF16 vs FP32
   - Different backends: vLLM vs SGLang vs FSDP vs Megatron
   - Different implementations even with identical weights

2. **Temporal Lag** (Model Staleness)
   - Rollout uses older checkpoint while training has progressed
   - Asynchronous rollout workers with stale parameters
   - Common in distributed/async RL systems

3. **Replay Buffers**
   - Training on historical trajectories from earlier iterations
   - Experience replay from different policy versions
   - Data augmentation or resampling strategies

4. **Off-Policy Algorithms**
   - Behavioral cloning from expert demonstrations
   - DAPO (data from auxiliary policies)
   - Any algorithm using trajectories from a different policy

5. **Data Quality Filtering**
   - Reweighting or filtering collected data
   - Preference learning with modified distributions
   - Curriculum learning with distribution shifts

These off-policy gaps can cause training instability and policy collapse. Rollout Correction uses importance sampling (IS) weights and rejection sampling (RS) to correct for any distribution shift between data collection and training.

**Important Note on Common Implementation Mistakes:**

Many LLM-RL implementations incorrectly apply PPO by **ignoring the actual rollout policy** π_rollout and assuming the training reference policy π_old is the behavior policy. This is mathematically incorrect when π_rollout ≠ π_old (which is typical in LLM-RL due to precision/backend differences between rollout and training).

**This is not PPO's fault** - PPO itself is mathematically correct. The issue is the incorrect assumption that π_old = π_rollout in naive implementations.

This critical implementation mistake that leads to RL training collapse was identified in the blog post ["When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"](https://richardli.xyz/rl-collapse) and motivated the development of this rollout correction framework.

**Mathematically correct approaches:**
- **Decoupled mode**: Three policies (π_rollout, π_old, π_θ) with IS correction from π_rollout to π_old
- **Bypass mode**: Two policies (π_rollout = π_old, π_θ) using actual rollout policy as PPO anchor
- **Bypass + Policy Gradient mode**: Two policies (π_rollout, π_θ) with IS/RS correction and no PPO clipping

See [Mathematical Formulations](rollout_corr_math.md#38-common-implementation-mistake) for detailed explanation.

### Key Design Principle: Separation of IS Weights and Rejection Sampling

The implementation cleanly separates two orthogonal mechanisms:

1. **IS Weights** (`rollout_is_weights`): Continuous reweighting for gradient correction
   - Policy ratio: π_old/π_rollout (decoupled) or π_θ/π_rollout (bypass)
   - **Safety-bounded**: Clamped to [exp(-20), exp(20)] ≈ [2e-9, 5e8] to prevent overflow
     * Token level: Bounds per-token ratios
     * Sequence level: Bounds product of ratios (broadcast to all tokens)
   - **Truncated**: Upper clamped via `.clamp(max=rollout_is_threshold)` (TIS: Truncated Importance Sampling)
   - **Zeroed at padding**: Multiplied by response_mask to zero out padding positions
   - Used to weight policy gradients (variance reduction)

2. **Rejection Sampling** (`modified_response_mask`): Binary filtering for outlier exclusion
   - Creates binary mask: 1 = keep, 0 = reject
   - Rejects tokens/sequences with IS ratios outside [lower_threshold, upper_threshold]
   - **Veto mechanism**: Independently rejects sequences containing catastrophic tokens
   - Modifies response_mask to exclude rejected samples from training
   - Used for loss aggregation (rejected samples don't contribute to gradients or denominators)

This separation ensures:
- ✅ IS weights provide continuous reweighting (reduce variance)
- ✅ Rejection sampling provides hard filtering (remove extreme outliers)
- ✅ Both mechanisms can be enabled independently or together
- ✅ Correct loss normalization (rejected samples excluded from all calculations)
- ✅ Safety bounds prevent numerical overflow in all cases

## Quick Start: Using Verified Presets

**NEW**: We now provide typed configuration with verified presets for common scenarios. These presets have been validated with tens of thousands of GPU hours across various models and training scenarios.

### Python API

```python
from verl.trainer.config.algorithm import RolloutCorrectionConfig

# === Decoupled PPO mode (3 policies: π_rollout, π_old, π_θ) ===
# IS weights correct for gap between π_old and π_rollout
config = RolloutCorrectionConfig.decoupled_token_is()   # Token-TIS
config = RolloutCorrectionConfig.decoupled_seq_is()     # Seq-TIS
config = RolloutCorrectionConfig.decoupled_seq_is_rs()  # Seq-MIS
config = RolloutCorrectionConfig.decoupled_geo_rs()     # Geo-RS
config = RolloutCorrectionConfig.geo_rs_seq_tis()       # Geo-RS-Seq-TIS

# === Bypass PPO mode (2 policies: π_rollout = π_old, π_θ) - fast ===
# PPO ratio handles IS, so no explicit IS weights needed
config = RolloutCorrectionConfig.bypass_ppo_clip()          # PPO-clip only
config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()   # PPO-clip + Geo-RS

# === Bypass PG mode (2 policies, no PPO clipping) - fast ===
# IS weights computed on-the-fly as π_θ / π_rollout
config = RolloutCorrectionConfig.bypass_pg_is()                # Seq-TIS + PG
config = RolloutCorrectionConfig.bypass_pg_rs()                # Geo-RS + PG
config = RolloutCorrectionConfig.bypass_pg_geo_rs_seq_tis()    # Geo-RS-Seq-TIS + PG

# === Other ===
config = RolloutCorrectionConfig.disabled()             # Metrics only (no correction)
```

### YAML Configuration (Advanced)

For advanced customization or YAML-based configs:

```yaml
algorithm:
  rollout_correction:
    rollout_is: token                      # IS weights: "token", "sequence", or null
    rollout_is_threshold: 2.0              # Upper threshold for IS weights
    rollout_is_batch_normalize: false      # Batch normalize IS weights to mean=1.0
    rollout_rs: null                       # Rejection sampling: "token", "sequence", "geometric", or null
    rollout_rs_threshold: null             # RS upper threshold (required if rollout_rs is enabled)
    rollout_rs_threshold_lower: null       # RS lower threshold (auto-reciprocal if null)
    rollout_token_veto_threshold: null     # Per-token veto threshold (null = disabled)
    bypass_mode: false  # Skip old_log_prob computation (sets π_old = π_rollout)
    loss_type: ppo_clip            # Loss type in bypass mode: "ppo_clip" (default) or "reinforce"

# REQUIRED: Enable log prob calculation
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
```

## Files

### **Core Implementation**

- `verl/trainer/ppo/rollout_corr_helper.py` - Contains `compute_rollout_correction_and_rejection_mask()` and `compute_offpolicy_metrics()`
- `verl/trainer/ppo/core_algos.py` - Rollout Correction integration with PPO and REINFORCE modes (`compute_policy_loss_bypass_mode()`, `compute_policy_loss_reinforce()`)
- `verl/trainer/ppo/ray_trainer.py` - Bypass mode implementation (skips `old_log_prob` computation)
- `verl/workers/actor/dp_actor.py` - Mode selection logic and metrics collection

### **Configuration Files**

- `verl/trainer/config/algorithm.py` - Rollout Correction parameters in `AlgoConfig`
- `verl/workers/config/actor.py` - Rollout Correction parameters in `ActorConfig`
- `verl/trainer/config/actor/actor.yaml` - Rollout Correction configuration section
- `verl/trainer/config/ppo_trainer.yaml` - Algorithm config with Rollout Correction

### **Documentation**

- `docs/examples/config.rst` - Configuration parameter descriptions

### **Example Scripts**

- `recipe/dapo/run_dapo_qwen2.5_32b_rollout_corr.sh` - DAPO example with Rollout Correction
- `examples/rollout_correction/run_with_rollout_corr.sh` - Basic example

### **Tests**

- `tests/trainer/ppo/test_rollout_corr.py` - Unit tests for IS/RS mechanisms
- `tests/trainer/ppo/test_rollout_corr_integration.py` - Integration tests

## Configuration Parameters

All parameters are under `algorithm.rollout_correction`:

### `rollout_is` (str or null)
Importance sampling weights aggregation level:
- `null` = No IS weights computed (metrics-only mode)
- `"token"`: Per-token IS weights
  - **Decoupled mode**: ρ_t = π_old(t)/π_rollout(t)
  - **Bypass/Pure IS mode**: ρ_t = π_θ(t)/π_rollout(t)
  - Independent truncation per token
  - Typical threshold: 1.5 - 5.0
- `"sequence"`: Per-sequence weight ρ_seq = ∏_t ρ_t
  - Multiplicative aggregation across sequence
  - Typical threshold: 2.0 - 10.0

All IS weights are safety-bounded to [exp(-20), exp(20)] ≈ [2e-9, 5e8]

### `rollout_is_threshold` (float)
Upper threshold for IS weight truncation. Default: `2.0`
- Truncates IS weights via `.clamp(max=rollout_is_threshold)` (TIS: Truncated Importance Sampling)
- Applied to IS weights for variance reduction
- Separate from rejection sampling (controlled by `rollout_rs` parameters)

### `rollout_rs` (str or null)
Rejection sampling aggregation level:
- `null` = No rejection sampling
- `"token"`: Reject individual tokens with outlier ratios
- `"sequence"`: Reject entire sequences with outlier ratios
- `"geometric"`: Geometric mean aggregation for rejection
  - Typical threshold: 1.0002 - 1.001

### `rollout_rs_threshold` (float or null)
Upper threshold for rejection sampling. Default: `null`
- **Required** when `rollout_rs` is enabled (must be explicitly set)
- Tokens/sequences with ratios > threshold are masked out

### `rollout_rs_threshold_lower` (float or null)
Lower threshold for rejection sampling. Default: `null`
- If `null`, uses reciprocal of upper threshold (1/upper)
- Tokens/sequences with ratios < threshold are masked out

### `rollout_token_veto_threshold` (float or null)
Per-token veto for catastrophic outliers. Default: `null`
- Checks **unclamped per-token ratios** before safety bounds
- If ANY token has ratio < threshold, entire sequence is rejected
- Independent of `rollout_is` and `rollout_rs` settings
- Typical values: `1e-4` to `1e-6` when enabled
- Example: `1e-4` catches tokens 10,000x less likely

### `rollout_is_batch_normalize` (bool)
Apply batch normalization to IS weights. Default: `False`
- `True`: Normalize IS weights to have mean=1.0 within each batch
  - **Token-level IS**: Normalizes over all token weights
  - **Sequence-level IS**: Normalizes over sequence means (one weight per sequence)
- `False`: Use raw (truncated) IS weights
- Reduces variance by ensuring average weight is 1.0 per batch
- Applied AFTER truncation to preserve truncation semantics
- Only affects IS weight values, not rejection sampling

## Understanding the Framework: Components and Combinations

The rollout correction framework is built from **orthogonal components** that can be combined flexibly. Understanding these components helps you choose the right configuration for your scenario.

### Key Components

1. **Operating Mode** (Section: [Operation Modes](#operation-modes))
   - **Decoupled**: Three policies (π_rollout, π_old, π_θ) with separate π_old computation
   - **Bypass**: Two policies (π_rollout = π_old, π_θ), skips π_old computation

2. **Loss Function** (in bypass mode, controlled by `loss_type`)
   - **PPO-clip** (`loss_type="ppo_clip"`, default): PPO clipped objective (IS handled by ratio)
   - **REINFORCE** (`loss_type="reinforce"`): Policy gradient with explicit IS weights (no clipping)

3. **IS/RS Aggregation Level**
   - **Token**: Per-token IS weights/rejection
   - **Sequence**: Sequence-level IS weights/rejection
   - **Geometric**: Geometric mean (for rejection only)

4. **Safety Mechanisms**
   - **Veto**: Rejects sequences with catastrophic tokens

See [Mathematical Formulations](rollout_corr_math.md#3-algorithmic-components-and-combinations) for detailed theory.

---

## Preset Configuration Guide

This section provides detailed guidance on choosing and using the verified presets. Each preset is a specific combination of components optimized for common scenarios.

### Understanding the Presets

#### Available Preset Methods

| Preset Method | Estimator | Mode | IS Level | RS Level | Properties |
|---------------|-----------|------|----------|----------|------------|
| **Decoupled PPO Mode** (3 policies: π_rollout, π_old, π_θ) |
| `decoupled_token_is()` | Token-TIS | Decoupled | token | - | Per-token IS weights |
| `decoupled_seq_is()` | Seq-TIS | Decoupled | sequence | - | Sequence-level IS weights |
| `decoupled_seq_is_rs()` | Seq-MIS | Decoupled | sequence | sequence | Sequence IS + sequence RS |
| `decoupled_geo_rs()` | Geo-RS | Decoupled | - | geometric + veto | Geometric RS + veto, no IS weights |
| `geo_rs_seq_tis()` | Geo-RS-Seq-TIS | Decoupled | sequence | geometric + veto | Geometric filter + clipped weight |
| **Bypass Mode (PPO-clip)** (2 policies; ratio handles IS, RS masks outliers) |
| `bypass_ppo_clip()` | - | Bypass (PPO-clip) | - | - | PPO-clip only |
| `bypass_ppo_clip_geo_rs()` | Geo-RS | Bypass (PPO-clip) | - | geometric + veto | PPO-clip + Geo-RS |
| **Bypass Mode (REINFORCE)** (2 policies; explicit IS weights, no PPO clipping) |
| `bypass_pg_is()` | Seq-TIS | Bypass (REINFORCE) | sequence | - | REINFORCE with explicit IS |
| `bypass_pg_rs()` | Geo-RS | Bypass (REINFORCE) | - | geometric + veto | REINFORCE with Geo-RS |
| `bypass_pg_geo_rs_seq_tis()` | Geo-RS-Seq-TIS | Bypass (REINFORCE) | sequence | geometric + veto | REINFORCE + Geo filter + seq IS |
| **Other** |
| `disabled()` | - | - | - | - | Metrics only, no correction |

**Note:**
- **Bypass mode** sets π_old = π_rollout and uses `loss_type` to select the loss function:
  - `"ppo_clip"` (default): PPO clipped objective where ratio = π_θ/π_rollout already handles IS
  - `"reinforce"`: REINFORCE with explicit IS weights as π_θ / π_rollout
- Both loss types benefit from rejection sampling (RS) which masks out-of-distribution samples.
- Estimators (Token-TIS, Seq-TIS, Seq-MIS, Geo-RS, Geo-RS-Seq-TIS) are compatible with Decoupled and Bypass modes.

#### Other Supported Combinations (Manual Configuration Required)

**Other supported combinations without preset methods:**
- Token IS + Token RS: Token-level IS weights + token-level RS mask
- Pure token RS: Token-level RS only, no IS weights
- Pure sequence RS: Sequence-level RS only, no IS weights

See [detailed configuration examples below](#additional-useful-configurations-not-exposed-as-presets) for manual configurations.

**Key properties:**
- Any aggregation level (token/sequence/geometric) works in either decoupled or bypass mode
- All combinations are fully supported by the implementation
- Rejection sampling is independent of IS weighting
- Pure RS (`bypass_pg_rs`) uses bypass + geometric RS with `loss_type="reinforce"` (no IS weights)

---

### 1. Decoupled Mode with Token-level Importance Sampling (`decoupled_token_is`)

**Configuration:**
```python
config = RolloutCorrectionConfig.decoupled_token_is(threshold=2.0)
```

**Components:**
- **Operating Mode**: Decoupled (3 policies)
- **Loss**: PPO with clipping (only for the second drift correction)
- **IS Aggregation**: Token-level
- **RS**: None (can be added separately)

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: null
    bypass_mode: false  # Decoupled mode
```

**Properties:**
- Independent truncation per token
- Lower variance than sequence-level (product of ratios bounded individually)
- Typical threshold: 1.5 - 5.0

**Theory:** See [rollout_corr_math.md §3.3.1](rollout_corr_math.md#331-token-level-aggregation)

---

### 2. Decoupled Mode with Sequence-level Importance Sampling (`decoupled_seq_is`)

**Also known as: Seq-TIS (Sequence-Level Truncated IS)**

**Configuration:**
```python
config = RolloutCorrectionConfig.decoupled_seq_is(threshold=2.0)
```

**Components:**
- **Operating Mode**: Decoupled (3 policies)
- **Loss**: PPO with clipping (only for the second drift correction)
- **IS Aggregation**: Sequence-level (Seq-TIS)
- **RS**: None (can be added separately)

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: null
    bypass_mode: false  # Decoupled mode
```

**Properties:**
- Multiplicative aggregation across sequence
- More sensitive to outliers than token-level
- Typical threshold: 2.0 - 10.0 (higher than token-level)

**Theory:** See [rollout_corr_math.md §3.3.2](rollout_corr_math.md#332-sequence-level-aggregation)

---

### 3. Decoupled Mode with Sequence-level IS + Rejection Sampling (`decoupled_seq_is_rs`)

**Also known as: Seq-MIS (Sequence-Level Masked IS)**

**Configuration:**
```python
config = RolloutCorrectionConfig.decoupled_seq_is_rs(is_threshold=2.0, rs_threshold=2.0)
```

**Components:**
- **Operating Mode**: Decoupled (3 policies)
- **Loss**: PPO with clipping (only for the second drift correction)
- **IS Aggregation**: Sequence-level (Seq-TIS)
- **RS**: Sequence-level rejection (Seq-MIS)

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: sequence
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0
    bypass_mode: false  # Decoupled mode
```

**Properties:**
- Double mechanism: IS reweighting (Seq-TIS) + rejection filtering (Seq-MIS)
- Lower effective sample size (rejects outliers)
- For severe off-policy gaps or when the distribution tail is "toxic" (garbage/adversarial samples)

**When to use Seq-MIS over Seq-TIS:**
- **Seq-TIS (clipping only)**: Maximizes information efficiency; extracts signal from all samples. Use when data is clean and mismatch is moderate.
- **Seq-MIS (rejection)**: Maximizes safety; acts as a hard trust region filter. Use when mismatch is severe or when high-weight samples are likely garbage rather than signal.

**Theory:** See [rollout_corr_math.md §3.4](rollout_corr_math.md#34-rejection-sampling-rs)

---

### 4. Decoupled Mode with Geometric Rejection Sampling (`decoupled_geo_rs`)

**Configuration:**
```python
config = RolloutCorrectionConfig.decoupled_geo_rs(rs_threshold=1.001, veto_threshold=1e-4)
```

**Components:**
- **Operating Mode**: Decoupled (3 policies)
- **Loss**: PPO with clipping (only for the second drift correction)
- **IS Aggregation**: None (pure rejection)
- **RS**: Geometric-level rejection (Geo-RS)
- **Veto**: Enabled

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: null
    rollout_rs: geometric
    rollout_rs_threshold: 1.001
    rollout_rs_threshold_lower: 0.999
    rollout_token_veto_threshold: 1e-4
    bypass_mode: false  # Decoupled mode
```

**Properties:**
- No IS weights (pure rejection)
- Geometric mean aggregation (more sensitive than arithmetic product)
- Typical threshold: 1.0001 - 1.001 (tighter than sequence/token level)
- Rejects sequences based on average per-token ratio deviation

**Why Geo-RS?** Standard IS estimators have a **Length Trap**: they penalize long sequences because the importance ratio grows exponentially with length. For reasoning models (CoT) and agents, this causes "Context Collapse" - the model learns from short answers while rejecting long chains of thought. Geo-RS normalizes by sequence length, making rejection length-invariant.

**Why tight thresholds?** Geometric mean is very sensitive. For 100 tokens with ratio 1.01 each:
- Product: 1.01^100 ≈ 2.7
- Geometric mean: 1.01

A threshold of 1.001 rejects sequences with average per-token deviation > 0.1%.

**Theory:** See [rollout_corr_math.md §3.3.3](rollout_corr_math.md#333-geometric-aggregation-geo-rs)

---

### 5. Geo-RS with Sequence IS (`geo_rs_seq_tis`)

**Also known as: Geo-RS-Seq-TIS**

**Configuration:**
```python
config = RolloutCorrectionConfig.geo_rs_seq_tis(
    is_threshold=2.0,
    rs_threshold=1.001,
    veto_threshold=1e-4
)
```

**Components:**
- **Operating Mode**: Decoupled (3 policies)
- **Loss**: PPO with clipping (only for the second drift correction)
- **IS Aggregation**: Sequence-level (Seq-TIS) for debiasing
- **RS**: Geometric-level rejection (Geo-RS) for length-invariant filtering
- **Veto**: Enabled

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: geometric
    rollout_rs_threshold: 1.001
    rollout_rs_threshold_lower: 0.999
    rollout_token_veto_threshold: 1e-4
    bypass_mode: false  # Decoupled mode
```

**Properties:**
- Combines **Geometric Filter** (length-invariant validity) with **Clipped Sequence Weight** (correct debiasing)
- Suitable for reasoning models (CoT, o1-style) and agents with long action sequences
- Solves the Length Trap while maintaining IS correction for bias reduction

**Theory:** See [rollout_corr_math.md §3.3.3](rollout_corr_math.md#333-geometric-aggregation-geo-rs)

---

### 6. Bypass Mode with PPO-clip (`bypass_ppo_clip`)

**Configuration:**
```python
config = RolloutCorrectionConfig.bypass_ppo_clip()
```

**Components:**
- **Operating Mode**: Bypass (2 policies: π_rollout = π_old, π_θ)
- **Loss**: PPO-clip (IS handled by ratio, no explicit IS weights)
- **IS Aggregation**: None (PPO ratio handles it)
- **RS**: None
- **Veto**: None

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: null
    rollout_rs: null
    bypass_mode: true
    loss_type: ppo_clip
```

**Properties:**
- PPO clipped objective in bypass mode
- The PPO ratio = π_θ/π_rollout already handles IS (no explicit IS weights needed)
- Skips `actor.compute_log_prob()` forward pass (2 policies instead of 3)
- No rejection sampling - use `bypass_ppo_clip_geo_rs()` for RS

**Configuration requirement:**
- Set `actor_rollout_ref.rollout.calculate_log_probs: true`

**Theory:** See [rollout_corr_math.md §3.1.2](rollout_corr_math.md#312-bypass-mode-two-policies)

---

### 6b. Bypass Mode with PPO-clip + Geo-RS (`bypass_ppo_clip_geo_rs`)

**Configuration:**
```python
config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs(
    rs_threshold=1.001,
    veto_threshold=1e-4
)
```

**Components:**
- **Operating Mode**: Bypass (2 policies: π_rollout = π_old, π_θ)
- **Loss**: PPO-clip (IS handled by ratio, no explicit IS weights)
- **IS Aggregation**: None (PPO ratio handles it)
- **RS**: Geometric-level rejection
- **Veto**: Enabled

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: null
    rollout_rs: geometric
    rollout_rs_threshold: 1.001
    rollout_rs_threshold_lower: 0.999
    rollout_token_veto_threshold: 1e-4
    bypass_mode: true
    loss_type: ppo_clip
```

**Properties:**
- PPO clipped objective in bypass mode with geometric RS
- The PPO ratio = π_θ/π_rollout already handles IS (no explicit IS weights needed)
- Skips `actor.compute_log_prob()` forward pass (2 policies instead of 3)
- Geometric RS masks outliers
- Veto mechanism enabled
- Solves Length Trap problem for CoT/agent workloads

**Configuration requirement:**
- Set `actor_rollout_ref.rollout.calculate_log_probs: true`

**Theory:** [§3.1.2 (Bypass)](rollout_corr_math.md#312-bypass-mode-two-policies) + [§3.3.3 (Geometric)](rollout_corr_math.md#333-geometric-aggregation-geo-rs)

---

### 7. REINFORCE with IS (`bypass_pg_is`)

**Configuration:**
```python
config = RolloutCorrectionConfig.bypass_pg_is(threshold=2.0)
```

**Components:**
- **Operating Mode**: Bypass (2 policies: π_rollout, π_θ)
- **Loss**: REINFORCE (policy gradient with explicit IS weights, no PPO clipping)
- **IS Aggregation**: Sequence-level
- **RS**: None

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: null
    bypass_mode: true
    loss_type: reinforce  # REINFORCE with explicit IS weights
```

**Properties:**
- REINFORCE loss with explicit IS weights (no PPO clipping)
- Single forward pass (skips old_log_prob computation)
- IS weights computed on-the-fly in loss function

**Theory:** See [rollout_corr_math.md §3.2.2](rollout_corr_math.md#322-policy-gradient-loss-with-isrs-correction)

---

### 8. REINFORCE with Rejection Sampling (`bypass_pg_rs`)

**Configuration:**
```python
config = RolloutCorrectionConfig.bypass_pg_rs(
    rs_threshold=1.001,
    veto_threshold=1e-4
)
```

**Components:**
- **Operating Mode**: Bypass (2 policies: π_rollout, π_θ)
- **Loss**: REINFORCE (no PPO clipping)
- **IS Aggregation**: None
- **RS**: Geometric-level rejection
- **Veto**: Enabled

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: null
    rollout_rs: geometric
    rollout_rs_threshold: 1.001
    rollout_rs_threshold_lower: 0.999
    rollout_token_veto_threshold: 1e-4
    bypass_mode: true
    loss_type: reinforce
```

**Properties:**
- Pure geometric RS (no IS weights, only rejection)
- Skips `actor.compute_log_prob()` forward pass (bypass mode)
- Veto mechanism enabled
- Typical threshold: 1.0001 - 1.001 (tighter than sequence/token level)

**Theory:** [§3.1.2 (Bypass)](rollout_corr_math.md#312-bypass-mode-two-policies) + [§3.3.3 (Geometric)](rollout_corr_math.md#333-geometric-aggregation-geo-rs)

---

### 9. REINFORCE with Geo-RS-Seq-TIS (`bypass_pg_geo_rs_seq_tis`)

**Also known as: Geo-RS-Seq-TIS in bypass mode**

**Configuration:**
```python
config = RolloutCorrectionConfig.bypass_pg_geo_rs_seq_tis(
    is_threshold=2.0,
    rs_threshold=1.001,
    veto_threshold=1e-4
)
```

**Components:**
- **Operating Mode**: Bypass (2 policies: π_rollout, π_θ)
- **Loss**: REINFORCE (no PPO clipping)
- **IS Aggregation**: Sequence-level (Seq-TIS)
- **RS**: Geometric-level rejection (Geo-RS)
- **Veto**: Enabled

**Equivalent YAML:**
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: geometric
    rollout_rs_threshold: 1.001
    rollout_rs_threshold_lower: 0.999
    rollout_token_veto_threshold: 1e-4
    bypass_mode: true
    loss_type: reinforce
```

**Properties:**
- Combines geometric filter + clipped sequence weight with REINFORCE loss
- Skips `actor.compute_log_prob()` forward pass (bypass mode)
- Suitable for reasoning models (CoT, o1-style) when you want bypass mode efficiency
- No PPO clipping - relies on IS/RS for stability

**Theory:** See [rollout_corr_math.md §3.3.3](rollout_corr_math.md#333-geometric-aggregation-geo-rs)

---

## Additional Useful Configurations (Not Exposed as Presets)

These configurations are **fully supported** but don't have convenience preset methods yet.

### 1. Token IS + Token RS (`token_is_rs`)

Token-level IS weights with token-level RS mask.

**Python:**
```python
config = RolloutCorrectionConfig(
    rollout_is="token",
    rollout_is_threshold=2.0,
    rollout_rs="token",
    rollout_rs_threshold=2.0,
)
```

**Properties:** Per-token IS weights + per-token RS mask.

### 2. Pure Token RS (`token_rs`)

Token-level RS only, no IS weights.

**Python:**
```python
config = RolloutCorrectionConfig(
    rollout_is=None,
    rollout_rs="token",
    rollout_rs_threshold=2.0,
)
```

**Properties:** Token-level RS mask, no IS reweighting.

### 3. Pure Sequence RS (`seq_rs`)

Sequence-level RS only, no IS weights.

**Python:**
```python
config = RolloutCorrectionConfig(
    rollout_is=None,
    rollout_rs="sequence",
    rollout_rs_threshold=2.0,
)
```

**Properties:** Sequence-level RS mask, no IS reweighting.

---

### Summary: How IS Weights are Processed

IS weights (`rollout_is_weights`) go through a fixed processing pipeline:

**Stage 1: Safety Bound (Prevent Overflow)**
- Token level: `exp(clamp(log_ratio, -20, 20))` per token → bounds each token to [2e-9, 5e8]
- Sequence level: `exp(clamp(sum(log_ratio), -20, 20))` → bounds product to [2e-9, 5e8], broadcast to all tokens

**Stage 2: Truncation (Reduce Variance)**
- `.clamp(max=rollout_is_threshold)` → caps weights at upper threshold (TIS: Truncated Importance Sampling)
- No lower truncation (preserves unbiasedness for small weights)

**Stage 3: Padding Zeroing (Correct Aggregation)**
- `weights * response_mask` → zeros out padding positions

**Stage 4: Optional Batch Normalization**
- If `rollout_is_batch_normalize=True`: Normalize weights to mean=1.0 within batch
- Applied after truncation to preserve truncation semantics

**Rejection Sampling (Separate Mechanism)**

Rejection sampling modifies `response_mask` (NOT weights) through `compute_rollout_rejection_mask()`:
- Computes safety-bounded ratios independently
- Creates binary mask: tokens/sequences outside [lower_threshold, upper_threshold] → 0 (rejected)
- Veto: Checks **unclamped per-token ratios** (before safety bound), rejects entire sequences containing catastrophic tokens
- Modified mask used for loss aggregation (rejected samples excluded from training)

## Operation Modes

The framework provides **two operating modes** for computing π_old, which can be combined with different loss functions.

### Operating Modes and Configuration

| Configuration | `bypass_mode` | `loss_type` | Operating Mode | Loss Function | Description |
|---------------|---------------|-------------|----------------|---------------|-------------|
| **Decoupled** | `false` | N/A | Decoupled | PPO | Computes `old_log_prob` separately via `actor.compute_log_prob()` |
| **Bypass + PPO-clip** | `true` | `"ppo_clip"` (default) | Bypass | PPO-clip | PPO clipped objective (IS handled by ratio) |
| **Bypass + REINFORCE** | `true` | `"reinforce"` | Bypass | REINFORCE | Policy gradient with explicit IS weights (no PPO clipping) |

### Operating Mode Details

#### Decoupled Mode (Three Policies)

**Policy setup:**
- π_rollout: Behavior policy (data collection)
- π_old: Proximal policy (computed via `actor.compute_log_prob()` at start of training epoch)
- π_θ: Current policy (being updated)

**Configuration:** `bypass_mode = false`

**Properties:**
- ✅ Achieves batch size invariance
- ✅ Separately corrects Drift 1 (rollout→old) and Drift 2 (old→current)
- ✅ Efficient stale data utilization
- ❌ Extra forward pass needed (`actor.compute_log_prob()`)

**Theory:** See [rollout_corr_math.md §3.1.1](rollout_corr_math.md#311-decoupled-mode-three-policies)

#### Bypass Mode (Two Policies)

**Policy setup:**
- π_rollout: Behavior policy (data collection)
- π_old = π_rollout: Proximal policy equals behavior policy
- π_θ: Current policy (being updated)

**Configuration:** `bypass_mode = true`

**Properties:**
- ✅ Skips `actor.compute_log_prob()` call (faster)
- ✅ Handles off-policy correction via IS/RS (when using policy gradient with IS/RS)
- ✅ Uses two policies instead of three (π_rollout = π_old)
- ⚠️ Does not separate proximal policy from behavior policy (unlike decoupled mode)

**Theory:** See [rollout_corr_math.md §3.1.2](rollout_corr_math.md#312-bypass-mode-two-policies)

---

### IS/RS Aggregation Levels (Orthogonal to Operating Mode)

The aggregation level can be chosen **independently** of the operating mode. Any aggregation level works in either decoupled or bypass mode.

| `rollout_is` | `rollout_rs` | Behavior |
|--------------|--------------|----------|
| `null` | `null` | **Disabled**: No computation, no metrics, no rejection |
| `null` | `"token"`, `"sequence"`, or `"geometric"` | **Rejection only**: Compute metrics, NO weight correction, YES rejection sampling |
| `"token"` or `"sequence"` | `null` | **IS weights only**: Weight correction enabled, NO rejection sampling |
| `"token"` or `"sequence"` | `"token"`, `"sequence"`, or `"geometric"` | **Full correction**: Both weight correction and rejection sampling enabled |

### Key Insights

- ✅ Any IS/RS aggregation level (token/sequence/geometric) can be used in **either** decoupled or bypass mode
- ✅ You can use **rejection sampling alone** without IS weight correction (`rollout_is=null, rollout_rs="token"`)
- ✅ You can use **IS weights alone** without outlier rejection (`rollout_is="token", rollout_rs=null`)
- ✅ You can use **both together** (`rollout_is="token", rollout_rs="token"`)
- ✅ You can **monitor metrics only** without any correction by setting both to `null` but still providing rollout_log_probs

**Veto rejection** (if enabled via `rollout_token_veto_threshold`) is applied **independently** of IS and RS settings.

**Theory:** See [rollout_corr_math.md §3.3](rollout_corr_math.md#33-isrs-aggregation-levels) for details on aggregation levels.

### Example Workflow

**Recommended: Bypass Mode**

This workflow uses bypass mode for efficiency.

1. **Start with metrics only** to understand the off-policy gap:
   ```yaml
   algorithm:
     rollout_correction:
       rollout_is: null
       rollout_rs: null
       bypass_mode: true  # Bypass mode (recommended)
       loss_type: ppo_clip  # Default: PPO clipped objective
   ```
   Monitor `rollout_corr/kl`, `rollout_corr/log_ppl_abs_diff`, `rollout_corr/chi2_token` to assess off-policy gap.

2. **Enable rejection sampling** if you see high outlier fractions:
   ```yaml
   algorithm:
     rollout_correction:
       rollout_is: null
       rollout_rs: sequence  # or "geometric" for higher sensitivity
       rollout_rs_threshold: 2.0
       bypass_mode: true  # Bypass mode
       loss_type: ppo_clip  # or "reinforce" for explicit IS weights
   ```
   This excludes outliers from training without modifying gradients.

3. **Enable full IS correction** (with REINFORCE loss) once comfortable with metrics:
   ```yaml
   algorithm:
     rollout_correction:
       rollout_is: sequence  # Recommended: unbiased, suitable for most cases
       rollout_is_threshold: 2.0
       rollout_rs: sequence  # or "geometric" for more aggressive filtering
       rollout_rs_threshold: 2.0
       bypass_mode: true  # Bypass mode
       loss_type: reinforce  # REINFORCE with explicit IS weights
   ```

**Benefits of bypass mode:**
- ✅ Skips expensive `actor.compute_log_prob()` forward pass (faster)
- ✅ `loss_type` controls the loss function: "ppo_clip" (default) or "reinforce"
- ✅ PPO-clip: IS handled by ratio (no explicit weights), RS mask applied
- ✅ REINFORCE: Explicit IS weights computed on-the-fly (π_θ / π_rollout)
- ✅ Both loss types work with all IS/RS combinations

## Usage

### Basic Setup

```yaml
algorithm:
  rollout_correction:
    rollout_is: token           # Enable IS weights at token level
    rollout_is_threshold: 2.0   # Threshold for IS weights
    rollout_rs: null            # No rejection sampling
    rollout_token_veto_threshold: null  # No veto

actor_rollout_ref:
  rollout:
    calculate_log_probs: true  # Required!
```

### Metrics

All metrics are prefixed with `rollout_corr/` in logs. For example, `rollout_is_mean` appears as `rollout_corr/rollout_is_mean`.

These metrics cover both:
- **Diagnostic metrics**: KL divergence, perplexity differences (measuring off-policy gap)
- **Correction statistics**: IS weights, rejection rates, veto stats (measuring correction applied)

#### **Core IS Weight Metrics**

- **`rollout_is_mean`**: Mean importance sampling weight across all valid tokens
  - Value close to 1.0 indicates minimal off-policy gap

- **`rollout_is_std`**: Standard deviation of IS weights
  - Higher values indicate greater variance in IS weights

- **`rollout_is_min`**: Minimum IS weight observed
  - Shows the most underweighted token/sequence
  - For sequence/geometric: computed from unclamped log-space ratios (true minimum)
  - For token: computed from safety-bounded weights

- **`rollout_is_max`**: Maximum IS weight observed
  - Shows the most overweighted token/sequence
  - For sequence/geometric: computed from unclamped log-space ratios (true maximum before safety bound)
  - For token: computed from safety-bounded weights (before threshold clamping)
  - Compare with `rollout_is_threshold` to see truncation impact

#### **Effective Sample Size**

- **`rollout_is_eff_sample_size`**: Effective sample size after IS weighting
  - **Formula**: `1 / mean(weights²)` where weights are normalized
  - **Range**: 0.0 to 1.0 (as fraction of original batch)
  - Lower values indicate weight concentration on fewer samples

#### **Veto Mechanism Metrics**

- **`rollout_is_veto_fraction`**: Fraction of sequences rejected by veto mechanism
  - **Important**: Sequences are rejected via `response_mask=0`, NOT by modifying IS weights
  - **IS weights unchanged by veto**: Already safety-bounded and truncated
  - Veto checks **unclamped per-token ratios** (true ratios before safety bound)
    - Decoupled mode: π_old(t)/π_rollout(t)
    - Bypass/Pure IS mode: π_θ(t)/π_rollout(t)
  - Detects catastrophic tokens (true ratio < veto_threshold, e.g., < 1e-4)

- **`rollout_is_catastrophic_token_fraction`**: Fraction of tokens below veto threshold
  - Identifies problematic tokens before sequence-level veto is applied
  - Checks **unclamped per-token ratios** (true ratios, not safety-bounded)
  - Each catastrophic token causes its entire sequence to be rejected

#### **Threshold Exceedance Metrics**

- **`rollout_is_ratio_fraction_high`**: Fraction of weights exceeding upper threshold
  - Shows how often truncation/masking occurs on high end
  - For sequence/geometric: computed from unclamped log-space ratios (true exceedance)
  - For token: computed from safety-bounded weights (before threshold clamping)

- **`rollout_is_ratio_fraction_low`**: Fraction of weights below lower threshold (1/upper_threshold)
  - Diagnostic metric showing how many weights are below the reciprocal threshold
  - For sequence/geometric: computed from unclamped log-space ratios (true exceedance)
  - For token: computed from safety-bounded weights (before truncation)

#### **Sequence-Level Metrics** (for sequence aggregation)

- **`rollout_is_seq_mean`**: Mean IS weight at sequence level
  - Should match `rollout_is_mean` for sequence-level aggregation

- **`rollout_is_seq_std`**: Standard deviation of sequence-level IS weights

- **`rollout_is_seq_min`**: Minimum sequence-level IS weight

- **`rollout_is_seq_max`**: Maximum sequence-level IS weight

- **`rollout_is_seq_max_deviation`**: Maximum absolute deviation from 1.0 at sequence level
  - Shows worst-case sequence off-policy gap

- **`rollout_is_seq_fraction_high`**: Fraction of sequences exceeding upper threshold

- **`rollout_is_seq_fraction_low`**: Fraction of sequences below lower threshold

#### **Rejection Sampling Metrics** (when `rollout_rs` is enabled)

- **`rollout_rs_masked_fraction`**: Fraction of tokens rejected via rejection sampling
  - **Important**: Rejection sampling modifies `response_mask` (sets rejected tokens to 0)
  - **Separate from IS weights**: IS weights are still truncated; rejection is an independent filtering step
  - Only present when `rollout_rs` is enabled (token/sequence/geometric)

- **`rollout_rs_seq_masked_fraction`**: Fraction of sequences with at least one rejected token
  - Shows sequence-level impact of rejection sampling
  - Token-level RS: sequence rejected if ANY token is outside [lower, upper]
  - Sequence-level RS: entire sequence rejected or accepted based on sequence-level ratio
  - Geometric RS: entire sequence rejected or accepted based on geometric mean

#### **Off-Policy Diagnostic Metrics** (Training vs Rollout Policy)

**Note on terminology:** These metrics use "training" to refer to the training reference policy and "rollout" to refer to π_rollout (the behavior policy used for data collection).
- **Decoupled mode**: "training" = π_old (computed at start of training epoch)
- **Bypass/Pure IS mode**: "training" = π_θ (current policy being trained)

In bypass/pure IS mode, metrics measure the drift between π_θ and π_rollout directly.

- **`training_ppl`**: Perplexity of training reference policy (π_old in decoupled mode, π_θ in bypass/pure IS mode)
  - **Formula**: `exp(-mean(log_probs))`
  - Lower values indicate higher model confidence

- **`rollout_ppl`**: Perplexity of rollout policy π_rollout (e.g., vLLM BF16)

- **`ppl_ratio`**: Ratio of training PPL to rollout PPL
  - **Formula**: `exp(mean(log(training_ppl / rollout_ppl)))`
  - **Meaning**: > 1.0 means training is less confident than rollout

- **`training_log_ppl`**: Log perplexity of training policy
  - Useful for identifying trends (linear scale)

- **`rollout_log_ppl`**: Log perplexity of rollout policy

- **`log_ppl_diff`**: Mean difference in log perplexities
  - **Formula**: `mean(log_ppl_rollout - log_ppl_training)`
  - Sign indicates which policy is more confident

- **`log_ppl_abs_diff`**: Mean absolute log perplexity difference
  - Magnitude of off-policy gap regardless of direction

- **`log_ppl_diff_max`**: Maximum log perplexity difference across sequences
  - Identifies worst-case sequence

- **`log_ppl_diff_min`**: Minimum log perplexity difference across sequences

- **`kl`**: KL divergence KL(π_rollout || π_training)
  - **Formula**: `mean(log_prob_rollout - log_prob_training)`
  - **Note**: Can be negative (rollout is less confident)

- **`k3_kl`**: K3 KL estimator
  - **Formula**: `mean(exp(log_ratio) - log_ratio - 1)`
  - More stable for small KL values
  - Always non-negative

- **`chi2_token`**: Chi-squared divergence at token level
  - **Formula**: `mean(ratio²) - 1` where ratio = π_training/π_rollout
  - Measures second moment of IS weight distribution
  - Always non-negative

- **`chi2_seq`**: Chi-squared divergence at sequence level
  - **Formula**: `mean((∏_t ratio_t)²) - 1`
  - Sequence-level second moment of IS weights
  - More sensitive than token-level chi-squared

#### **Example: Accessing Metrics in Code**

```python
# Metrics are returned from compute_rollout_correction_and_rejection_mask
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

# Returns 3 values (weights, modified_response_mask, metrics)
weights_proto, modified_response_mask, metrics = compute_rollout_correction_and_rejection_mask(
    old_log_prob=training_log_probs,      # from training policy
    rollout_log_prob=rollout_log_probs,   # from rollout policy
    response_mask=response_mask,
    rollout_is="token",  # Enable IS weights at token level
    rollout_is_threshold=2.0,
    rollout_rs="token",  # Enable rejection sampling at token level
    rollout_rs_threshold=2.0,
    rollout_rs_threshold_lower=0.5,
    rollout_token_veto_threshold=1e-4,  # Enable veto for catastrophic outliers
)

# Extract IS weights (processed, zeroed at padding)
is_weights = weights_proto.batch["rollout_is_weights"]

# IS weights processing (with IS enabled at token level):
# 1. Safety-bounded: exp(clamp(log_ratio, -20, 20)) per token
# 2. Truncated: .clamp(max=2.0) to cap extreme weights
# 3. Zeroed at padding positions
# Note: Truncation is ALWAYS applied to IS weights (TIS: Truncated Importance Sampling)

# modified_response_mask has rejection applied (since rollout_rs="token"):
# 1. RS rejection: tokens outside [0.5, 2.0] masked to 0 via response_mask
# 2. Veto rejection: sequences with catastrophic tokens (ratio < 1e-4) masked to 0
# Note: Veto checks unclamped per-token ratios (before safety bounds)
# Note: RS and IS are separate mechanisms - both can be enabled independently

# All metrics have 'rollout_corr/' prefix
print(f"Mean IS weight: {metrics['rollout_corr/rollout_is_mean']:.3f}")
print(f"Effective sample size: {metrics['rollout_corr/rollout_is_eff_sample_size']:.3f}")
print(f"Veto fraction: {metrics['rollout_corr/rollout_is_veto_fraction']:.3f}")
print(f"RS masked fraction: {metrics['rollout_corr/rollout_rs_masked_fraction']:.3f}")
print(f"KL divergence: {metrics['rollout_corr/kl']:.3f}")

# Check IS weights for valid tokens (non-padding)
valid_weights = is_weights[response_mask.bool()]
print(f"\n✓ IS weights min (valid tokens): {valid_weights.min():.4f}")
print(f"✓ IS weights max (valid tokens): {valid_weights.max():.4f}")
print(f"✓ All valid IS weights > 0: {(valid_weights > 0).all()}")
print(f"✓ IS weights are capped at threshold: {(valid_weights <= 2.0).all()}")

# Check rejection via response_mask
rejected_tokens = (response_mask == 1) & (modified_response_mask == 0)
print(f"\n✓ Rejected {rejected_tokens.sum()} tokens via response_mask")
print(f"✓ Rejection sampling modifies response_mask (separate from IS weight truncation)")
print(f"✓ IS weights are always truncated to [0, threshold] after safety bounding")

# Check for warning conditions
if metrics['rollout_corr/rollout_is_mean'] < 0.5 or metrics['rollout_corr/rollout_is_mean'] > 2.0:
    print("⚠️  Warning: Mean IS weight far from 1.0, significant off-policy gap detected")

if metrics['rollout_corr/rollout_is_eff_sample_size'] < 0.3:
    print("⚠️  Warning: Low effective sample size, high weight concentration")

if metrics['rollout_corr/rollout_is_veto_fraction'] > 0.1:
    print("⚠️  Warning: High veto fraction, policies may be too different")
```

#### **Example: Monitoring Metrics During Training**

```python
# In your training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # ... rollout phase ...

        # Compute IS weights and get metrics
        rollout_corr_config = config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None:
            weights_proto, modified_response_mask, metrics = compute_rollout_correction_and_rejection_mask(
                old_log_prob=batch.old_log_prob,
                rollout_log_prob=batch.rollout_log_prob,
                response_mask=batch.response_mask,
                rollout_is=rollout_corr_config.get("rollout_is", None),
                rollout_is_threshold=rollout_corr_config.get("rollout_is_threshold", 2.0),
                rollout_rs=rollout_corr_config.get("rollout_rs", None),
                rollout_rs_threshold=rollout_corr_config.get("rollout_rs_threshold", None),
                rollout_rs_threshold_lower=rollout_corr_config.get("rollout_rs_threshold_lower", None),
                rollout_token_veto_threshold=rollout_corr_config.get("rollout_token_veto_threshold", None),
            )

        # Log to tensorboard/wandb
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step=global_step)

        # IMPORTANT: Update batch response_mask with rejection applied
        batch.response_mask = modified_response_mask

        # Use IS weights in training (always safety-bounded, zeroed at padding)
        is_weights = weights_proto.batch["rollout_is_weights"]
        # ... apply weights to policy gradient ...
```

#### **Example: Conditional Alerting Based on Metrics**

```python
def check_rollout_correction_health(metrics, config):
    """Check if Rollout Correction metrics indicate healthy training."""
    warnings = []

    # Check mean IS weight
    mean_weight = metrics['rollout_corr/rollout_is_mean']
    if mean_weight < 0.5 or mean_weight > 2.0:
        warnings.append(f"Mean IS weight {mean_weight:.3f} is far from 1.0")

    # Check effective sample size
    ess = metrics['rollout_corr/rollout_is_eff_sample_size']
    if ess < 0.3:
        warnings.append(f"Effective sample size {ess:.3f} is too low")

    # Check veto fraction
    veto_frac = metrics['rollout_corr/rollout_is_veto_fraction']
    if veto_frac > 0.1:
        warnings.append(f"Veto fraction {veto_frac:.3f} is too high")

    # Check standard deviation
    std = metrics['rollout_corr/rollout_is_std']
    if std > 1.0:
        warnings.append(f"IS weight std {std:.3f} is too high")

    # Check KL divergence
    kl = metrics['rollout_corr/kl']
    if abs(kl) > 0.1:
        warnings.append(f"KL divergence {kl:.3f} indicates significant off-policy gap")

    # Check chi-squared divergence
    if 'rollout_corr/chi2_token' in metrics:
        chi2_token = metrics['rollout_corr/chi2_token']
        if chi2_token > 1.0:
            warnings.append(f"Chi-squared divergence (token) {chi2_token:.3f} indicates severe distribution shift")

    if warnings:
        print("⚠️  Rollout Correction Health Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        return False
    else:
        print("✅ Rollout Correction metrics look healthy")
        return True

# Use in training
_, _, metrics = compute_rollout_correction_and_rejection_mask(...)
is_healthy = check_rollout_correction_health(metrics, config)

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
bash examples/rollout_correction/run_with_rollout_corr.sh
```

Monitor metrics for 1-2 epochs before adjusting parameters.

## Configuration Examples

### Example 1: IS Weights Only (Token Level)
```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: null  # No rejection sampling
```

### Example 2: Rejection Sampling Only (No IS Weights)
```yaml
algorithm:
  rollout_correction:
    rollout_is: null  # No IS weights
    rollout_rs: token
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5
```

### Example 3: Both IS and RS (Geometric RS)
```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: geometric
    rollout_rs_threshold: 1.0002
    rollout_rs_threshold_lower: 0.9998
```

### Example 4: Full Correction with Veto
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence
    rollout_is_threshold: 2.0
    rollout_rs: token
    rollout_rs_threshold: 2.0
    rollout_rs_threshold_lower: 0.5
    rollout_token_veto_threshold: 1e-4  # Veto catastrophic tokens
```

### Example 5: Bypass Mode with PPO-clip (Default)
```yaml
algorithm:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
    rollout_rs: token
    rollout_rs_threshold: 2.0
    bypass_mode: true   # Skip old_log_prob computation
    loss_type: ppo_clip # PPO clipped objective (default)
```
**Skips expensive `actor.compute_log_prob()` forward pass. PPO ratio = π_θ/π_rollout handles IS.**

### Example 6: Bypass Mode with REINFORCE
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence                   # Explicit IS correction in loss
    rollout_is_threshold: 2.0
    rollout_rs: null                       # Optional: can add rejection sampling
    bypass_mode: true
    loss_type: reinforce           # REINFORCE with explicit IS weights
```
**No PPO clipping, pure policy gradient with IS correction**

### Example 7: Bypass Mode with PPO-clip + Rejection Sampling
```yaml
algorithm:
  rollout_correction:
    rollout_is: sequence                   # Computed for metrics
    rollout_is_threshold: 2.0
    rollout_rs: geometric                  # Rejection sampling enabled
    rollout_rs_threshold: 1.001
    bypass_mode: true
    loss_type: ppo_clip            # PPO clipped objective (IS handled by ratio)
```
**PPO clipping with rejection sampling. IS handled by PPO ratio (no explicit IS weights).**

## Troubleshooting

### Issue: High spread in IS weights
**Symptoms:** `rollout_is_std` > 1.0, `rollout_is_eff_sample_size` < 0.3

**Solutions:**
1. Switch from `sequence` to `geometric` level
2. Tighten thresholds
3. Verify rollout and training aren't too different

### Issue: Too many sequences vetoed
**Symptoms:** `rollout_is_veto_fraction` > 0.1

**Solutions:**
1. Relax veto threshold in config:
   ```yaml
   algorithm:
     rollout_correction:
       rollout_token_veto_threshold: 1e-3
   ```
2. Check for numerical issues in log prob computation
3. Verify policies aren't completely different

### Issue: Mean IS weight far from 1.0
**Symptoms:** `rollout_is_mean` < 0.5 or > 2.0

**Solutions:**
1. Verify `calculate_log_probs=True` is set
2. Check rollout_log_probs are correctly passed
3. Check for systematic distribution shift

### Debugging: Visualizing Metrics

**Example: Plot IS weight distribution**

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_is_metrics(metrics_history):
    """Plot rollout IS metrics over training steps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Mean IS weight over time
    axes[0, 0].plot(metrics_history['rollout_corr/rollout_is_mean'])
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Ideal')
    axes[0, 0].set_title('Mean IS Weight')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].legend()

    # Plot 2: Effective sample size
    axes[0, 1].plot(metrics_history['rollout_corr/rollout_is_eff_sample_size'])
    axes[0, 1].axhline(y=0.5, color='g', linestyle='--', label='Good')
    axes[0, 1].axhline(y=0.3, color='r', linestyle='--', label='Warning')
    axes[0, 1].set_title('Effective Sample Size')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].legend()

    # Plot 3: Veto fraction
    axes[0, 2].plot(metrics_history['rollout_corr/rollout_is_veto_fraction'])
    axes[0, 2].axhline(y=0.1, color='r', linestyle='--', label='Warning')
    axes[0, 2].set_title('Veto Fraction')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].legend()

    # Plot 4: KL divergence over time
    axes[1, 0].plot(metrics_history['rollout_corr/kl'], label='KL')
    axes[1, 0].plot(metrics_history['rollout_corr/k3_kl'], label='K3 KL')
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend()

    # Plot 5: PPL ratio over time
    axes[1, 1].plot(metrics_history['rollout_corr/ppl_ratio'])
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Ideal')
    axes[1, 1].set_title('PPL Ratio (Training/Rollout)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].legend()

    # Plot 6: Chi-squared divergence
    if 'rollout_corr/chi2_token' in metrics_history:
        axes[1, 2].plot(metrics_history['rollout_corr/chi2_token'], label='Token-level')
        if 'rollout_corr/chi2_seq' in metrics_history:
            axes[1, 2].plot(metrics_history['rollout_corr/chi2_seq'], label='Seq-level')
        axes[1, 2].axhline(y=1.0, color='r', linestyle='--', label='Warning')
        axes[1, 2].set_title('Chi-squared Divergence')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].legend()
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('rollout_is_metrics.png', dpi=150)
    print("Saved plot to rollout_is_metrics.png")
```

**Example: Metric collection during training**

```python
# Collect metrics over time
metrics_history = {
    'rollout_corr/rollout_is_mean': [],
    'rollout_corr/rollout_is_eff_sample_size': [],
    'rollout_corr/rollout_is_veto_fraction': [],
    'rollout_corr/kl': [],
    'rollout_corr/k3_kl': [],
    'rollout_corr/ppl_ratio': [],
    'rollout_corr/chi2_token': [],
    'rollout_corr/chi2_seq': [],
}

# In training loop
for step in range(num_steps):
    # ... compute IS weights and rejection mask ...
    _, _, metrics = compute_rollout_correction_and_rejection_mask(...)

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
- **Training stability**: Significantly improved when off-policy gap exists


## Testing

Run the test suite to verify everything works:

```bash
# Basic unit tests
python test_rollout_corr.py

# Integration tests (if pytest is available)
pytest tests/trainer/ppo/test_rollout_corr_integration.py -v
```

Expected output: All tests pass ✓

## Additional Resources

- **Implementation**: `verl/trainer/ppo/rollout_corr_helper.py`
- **Examples**: `examples/rollout_correction/`
- **DAPO Example**: `recipe/dapo/run_dapo_qwen2.5_32b_rollout_corr.sh`

## Summary

Rollout Correction provides a unified framework for handling general off-policy problems in RL:
- ✅ Corrects ANY distribution shift between data collection and training
- ✅ Supports diverse scenarios: policy mismatch, staleness, replay buffers, off-policy algorithms
- ✅ Numerical stability with safety bounds and rejection mechanisms
- ✅ Comprehensive diagnostics: KL, perplexity, χ² divergence
- ✅ Flexible methods from token-level to sequence-level aggregation
- ✅ Memory-efficient implementation

## References

- **[Mathematical Formulations](rollout_corr_math.md)** - Detailed mathematical theory and derivations for all rollout correction methods
- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse) (see Blog Series above for parts 1-3)
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)