# Mathematical Formulations of Rollout Correction Methods in `verl`

**Author:** Yingru Li
**Date:** 2025-11-03

---

## Abstract

This document provides the definitive mathematical formulations for all rollout correction methods implemented in the `verl` library. These methods provide a **unified framework for general off-policy reinforcement learning**, addressing any scenario where the data collection distribution differs from the training distribution.

**Applicable scenarios include:**
- **Policy mismatch**: Different precision (BF16 vs FP32), different backends (vLLM vs PyTorch)
- **Temporal lag**: Model staleness, asynchronous rollout workers
- **Replay buffers**: Training on historical trajectories from earlier policy versions
- **Off-policy algorithms**: Behavioral cloning, DAPO, expert demonstrations
- **Data filtering**: Reweighting, preference learning, curriculum learning

We detail the "three-policy problem" ($\pi_{\text{rollout}}$, $\pi_{\text{old}}$, $\pi_{\theta}$) and formalize the implementation of standard PPO, hybrid PPO+IS methods (token-level and sequence-level), high-speed bypass modes, and rejection sampling techniques. Finally, we document the diagnostic metrics used to monitor off-policy severity, such as $\chi^2$-divergence and KL divergence.

---

## Table of Contents

1. [The Three-Policy Problem and Notation](#the-three-policy-problem-and-notation)
2. [Standard PPO (Baseline)](#standard-ppo-baseline)
3. [Pure IS Methods (No PPO Clipping)](#pure-is-methods-no-ppo-clipping)
4. [PPO + IS Hybrid Methods](#ppo--is-hybrid-methods)
5. [Rejection Sampling Methods](#rejection-sampling-methods)
6. [Veto Mechanism (Catastrophic Outlier Protection)](#veto-mechanism-catastrophic-outlier-protection)
7. [Off-Policy Diagnostic Metrics](#off-policy-diagnostic-metrics)
8. [Summary Table](#summary-table)
9. [Implementation References](#implementation-references)

---

## The Three-Policy Problem and Notation

In general off-policy RL, we must manage three distinct policies that create two sources of distribution drift. The rollout correction framework handles **any scenario** where the data collection distribution differs from the training distribution.

### The Three Policies

**$\pi_{\text{rollout}}$ (Rollout/Behavior Policy)**
The policy used for data collection. This represents the behavior distribution $\mu$ from classical off-policy RL theory. In general off-policy learning, this can be ANY distribution that differs from the training policy.

- **When created**: During data collection phase
- **Purpose**: Generate trajectories for training (from ANY source)
- **Common sources in practice**:
  1. **Policy mismatch**: Same model weights, different implementation
     - Different precision: BF16 rollout vs FP32 training
     - Different backends: vLLM vs PyTorch/FSDP
  2. **Temporal lag**: Stale model checkpoint
     - Asynchronous rollout workers using older parameters
     - Distributed systems with parameter staleness
  3. **Replay buffers**: Historical data
     - Experience replay from earlier training iterations
     - Trajectories from multiple policy versions
  4. **Off-policy algorithms**: Different policy entirely
     - Behavioral cloning from expert demonstrations
     - DAPO (auxiliary policies)
     - Any external data source
  5. **Data filtering**: Modified distribution
     - Reweighted samples
     - Preference-filtered data
     - Curriculum learning with distribution shifts
- **Fixed**: Never changes during training on a batch (frozen behavior distribution)

**$\pi_{\text{old}}$ (Old Policy)**
The reference policy for PPO clipping. **Note: This is optional** - if `rollout_log_prob` is available, you can skip computing this separately.

- **When created**:
  - **Standard mode (optional)**: Computed at start of training epoch via `actor.compute_log_prob()` to separately track rollout→old drift
  - **Bypass mode (recommended)**: Reused from $\pi_{\text{rollout}}$ (zero-cost, assumes rollout ≈ old)
- **Purpose**: Anchor point for PPO clipping
- **Fixed**: Frozen during all PPO epochs on same batch
- **Key insight**: Computing `old_log_prob` separately is **not necessary** if you're willing to clip directly against the rollout policy

**$\pi_{\theta}$ (Current Policy)**
The training policy being actively optimized. This represents the target distribution $\pi$ from theory.

- **When created**: Continuously updated during training
- **Purpose**: The policy being optimized
- **Updated**: Every gradient step

### The Two Distribution Shifts

**Drift 1: $\pi_{\text{rollout}} \to \pi_{\text{old}}$** (The Off-Policy Gap - Optional to correct)
This is the **fundamental off-policy gap** that arises whenever the data collection distribution differs from the training distribution. This is the core problem that rollout correction methods address.

- **Nature**: Can range from negligible (similar implementations) to severe (completely different policies)
- **Examples in practice**:
  - **Negligible**: Same checkpoint, minor implementation differences (BF16 vs FP32)
  - **Moderate**: Model staleness from async workers (policy a few checkpoints old)
  - **Severe**: Replay buffers with old data, behavioral cloning from experts, DAPO with auxiliary policies
- **Correction** (if desired): $w = \pi_{\text{old}} / \pi_{\text{rollout}}$ via Importance Sampling
- **Optional**: Many users skip this correction (bypass mode) when the gap is negligible, setting $w = 1$
- **Key insight**: The severity of this drift determines which correction method to use (or whether to use any correction at all)

**Drift 2: $\pi_{\text{old}} \to \pi_{\theta}$** (On-Policy Drift)
This is the standard on-policy drift that occurs as $\pi_{\theta}$ is updated over multiple gradient steps. It is corrected by **PPO clipping**, independent of the off-policy gap.

- **Source**: Policy updates via gradient descent during training
- **Correction**: PPO clips ratio $r = \pi_{\theta} / \pi_{\text{old}}$ to prevent overly large updates
- **Universal**: Applies to both on-policy and off-policy training

### Notation

- $\pi_{\text{rollout}}$: Rollout policy (data collection)
- $\pi_{\text{old}}$: Old policy (PPO anchor)
- $\pi_{\theta}$: Current training policy (being updated)
- $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$: Per-token IS ratio (Corrects Drift 1)
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: PPO policy ratio (Corrects Drift 2)
- $A_t$: Advantage at token $t$
- $T$: Set of valid tokens in a sequence
- $\tau$: Upper threshold for IS weights (e.g., 2.0)
- $\tau_{\text{lower}}$: Lower threshold for IS weights (typically $1/\tau$)
- $\epsilon$: PPO clip range (typically 0.2)

---

## Standard PPO (Baseline)

The baseline PPO formulation only corrects for Drift 2 (on-policy drift) and **ignores any off-policy gap** between $\pi_{\text{rollout}}$ and $\pi_{\text{old}}$. This is appropriate when the off-policy gap is negligible or when no rollout policy log-probs are available.

### Loss Function

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: Policy ratio (current vs old policy)
- $\epsilon$: Clip range (typically 0.2)

**Implementation:** `compute_policy_loss()` in [core_algos.py](verl/trainer/ppo/core_algos.py#L812-L884)

---

## Pure IS Methods (No PPO Clipping)

### Method: `pure_is` (Pure Importance Sampling)

This method discards the PPO framework entirely and implements a pure policy gradient (REINFORCE) with IS correction for the off-policy gap. It uses bypass mode, meaning $\pi_{\text{old}}$ is not used. This is appropriate for **any off-policy scenario** where unbiased gradient estimates are required.

**Configuration:**
```python
RolloutCorrectionConfig.pure_is(threshold=2.0)
# or equivalently:
RolloutCorrectionConfig(
    rollout_is="sequence",
    rollout_is_threshold=2.0,
    rollout_rs=None,
    bypass_old_logprob_for_rollout=True,
    use_pure_rollout_correction=True,
)
```

#### Loss Function

$$
L_{\text{PureIS}}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ w_{\text{seq}} \cdot \sum_{t \in T} \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

The gradient is therefore:

$$
\nabla_\theta L_{\text{PureIS}} = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ w_{\text{seq}} \cdot \sum_{t \in T} \nabla_\theta \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

where the sequence-level IS weight $w_{\text{seq}}$ is computed on-the-fly and directly compares $\pi_{\theta}$ to $\pi_{\text{rollout}}$:

$$
w_{\text{seq}} = \min\left( \prod_{t \in T} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, \tau \right)
$$

#### Properties

- **Unbiased**: An unbiased estimator (when $\tau = \infty$) of the policy gradient
- **No PPO Clipping**: Does not use the PPO clipping objective
- **High Variance**: Suffers from both sequence-level IS variance and no clipping safety net
- **Fast**: Bypasses the $\pi_{\text{old}}$ log-prob computation (faster training)
- **On-the-fly IS**: Computes IS weights during actor forward pass

**Implementation:** `compute_policy_loss_with_rollout_correction()` in [core_algos.py](verl/trainer/ppo/core_algos.py#L1537-L1681)

---

## PPO + IS Hybrid Methods

These methods combine PPO clipping (for on-policy drift) with IS weight correction (for the off-policy gap). They provide the most comprehensive correction for **general off-policy scenarios** where both distribution shifts need to be addressed.

### Method: `token_is` / `token_tis` (Token-level Truncated IS)

This is the most common, low-variance correction for general off-policy problems, implementing a per-token version of Truncated Importance Sampling (TIS). **Recommended as the default method** for most off-policy scenarios with moderate distribution shift.

**Configuration:**
```python
RolloutCorrectionConfig.token_is(threshold=2.0)
# or equivalently:
RolloutCorrectionConfig(
    rollout_is="token",
    rollout_is_threshold=2.0,
    rollout_rs=None,
)
```

#### Objective Function

The PPO objective $J(\theta)$ is modified by multiplying the loss term by a per-token IS weight $w_t$.

$$
J_{\text{PPO+TIS}}(\theta) = \mathbb{E}_t \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- Per-token IS weight: $w_t = \min(\rho_t, \tau) = \min\left(\frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, \tau\right)$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- $\pi_{\text{old}}$ is computed by actor at **start of training epoch** (via `actor.compute_log_prob()`)

#### Properties

- **Biased but Low Variance**: Per-token truncation is stable
- **Double Correction**: $w_t$ corrects Drift 1, $r_t(\theta)$ is clipped for Drift 2
- **Standard Mode**: Computes $\pi_{\text{old}}$ separately. Slower, but most accurate

**Implementation:**
- IS weights: `compute_rollout_correction_weights()` in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L325-L402)
- PPO loss: `compute_policy_loss()` in [core_algos.py](verl/trainer/ppo/core_algos.py#L812-L884) with weights applied at line 967

---

### Method: `seq_is` (Sequence-level IS)

This is the unbiased (but high-variance) version of the hybrid method.

**Configuration:**
```python
RolloutCorrectionConfig.seq_is(threshold=2.0)
# or equivalently:
RolloutCorrectionConfig(
    rollout_is="sequence",
    rollout_is_threshold=2.0,
    rollout_rs=None,
)
```

#### Objective Function

The objective is identical to `token_is`, but uses a sequence-level weight $w_{\text{seq}}$ broadcast to all tokens.

$$
J_{\text{PPO+SeqIS}}(\theta) = \mathbb{E}_t \left[ w_{\text{seq}} \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- Sequence-level IS weight (broadcast to all tokens):
$$w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, \tau \right) = \min\left( \exp\left(\sum_{t \in T} \log \rho_t\right), \tau \right)$$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- $\pi_{\text{old}}$ is computed by actor at **start of training epoch**

#### Properties

- **Unbiased** (when $\tau = \infty$)
- **High Variance**: Uses the exponential product of ratios
- **Double Correction**: $w_{\text{seq}}$ corrects Drift 1, $r_t(\theta)$ is clipped for Drift 2
- **Standard Mode**: Computes $\pi_{\text{old}}$ separately. Slower, but most accurate

---

### Method: `ppo_is_bypass` (PPO + IS in Bypass Mode)

This method prioritizes speed by assuming $\pi_{\text{old}} = \pi_{\text{rollout}}$.

**Configuration:**
```python
RolloutCorrectionConfig.ppo_is_bypass(threshold=2.0)
# or equivalently:
RolloutCorrectionConfig(
    rollout_is="token",
    rollout_is_threshold=2.0,
    rollout_rs=None,
    bypass_old_logprob_for_rollout=True,
    use_pure_rollout_correction=False,
)
```

#### Key Insight

When `bypass_old_logprob_for_rollout=True`, we set $\pi_{\text{old}} = \pi_{\text{rollout}}$. This causes two things to happen:

1. The IS weight $w_t$ becomes 1:

$$
w_t = \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}} \to \frac{\pi_{\text{rollout}}}{\pi_{\text{rollout}}} = 1
$$

2. The PPO ratio $r_t$ changes its anchor:

$$
r_t(\theta) = \frac{\pi_{\theta}}{\pi_{\text{old}}} \to \frac{\pi_{\theta}}{\pi_{\text{rollout}}}
$$

This method degenerates to standard PPO, but clips directly against the rollout policy.

#### Objective Function

$$
J_{\text{PPO-Bypass}}(\theta) = \mathbb{E}_t \left[ 1.0 \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$.

#### Properties

- **Faster**: Bypasses the $\pi_{\text{old}}$ log-prob computation (significant speedup)
- **No IS Correction**: The $w_t$ weight becomes an identity
- **Single Correction**: PPO clips the total (rollout $\to$ current) drift directly
- **Trade-off**: Speed vs granular control

---

## Rejection Sampling Methods

These methods filter outlier sequences/tokens (by zeroing them out) instead of re-weighting them.

### Method: `seq_is_rs` / `seq_mis` (Mixture Importance Sampling)

This method combines sequence-level IS weighting with sequence-level rejection.

**Configuration:**
```python
RolloutCorrectionConfig.seq_is_rs(
    is_threshold=2.0,
    rs_threshold=2.0,
    rs_threshold_lower=None,  # defaults to 1/rs_threshold
)
# Alias:
RolloutCorrectionConfig.seq_mis(threshold=2.0)
```

#### Loss Function

The PPO loss is multiplied by both the IS weight $w_{\text{seq}}$ and a binary acceptance mask $\mathbb{1}_{\text{accept}}$.

$$
L_{\text{PPO+MIS}}(\theta) = -\mathbb{E}_t \left[ w_{\text{seq}} \cdot \mathbb{1}_{\text{accept}} \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, \tau_{\text{IS}} \right)$ (sequence-level IS weight)
- (rejection mask)

$$
\mathbb{1}_{\text{accept}} = \begin{cases} 1 & \text{if } \tau_{\text{lower}} \leq \prod_{t \in T} \rho_t \leq \tau_{\text{upper}} \\ 0 & \text{otherwise} \end{cases}
$$

#### Properties

- **Unbiased**: Rejects outliers rather than truncating
- **Lower Effective Sample Size**: Rejects entire sequences
- **Combines IS + RS**: Double correction mechanism

**Implementation:** `compute_rollout_rejection_mask()` in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L80-L188)

---

### Method: `geo_rs` / `geo_mis` (Geometric Rejection Sampling)

This method uses only rejection, based on the highly sensitive geometric mean of the ratios.

**Configuration:**
```python
RolloutCorrectionConfig.geo_rs(
    rs_threshold=1.001,  # Very tight threshold (±0.1%)
    rs_threshold_lower=None,
    veto_threshold=1e-4,
)
```

#### Loss Function

$$
L_{\text{GeoRS}}(\theta) = -\mathbb{E}_t \left[ \mathbb{1}_{\text{geo}} \cdot \mathbb{1}_{\text{veto}} \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- Geometric mean: $\rho_{\text{geo}} = \exp\left( \frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{1/|T|}$
- Geometric acceptance: $\mathbb{1}_{\text{geo}} = \begin{cases} 1 & \text{if } \tau_{\text{lower}} \leq \rho_{\text{geo}} \leq \tau_{\text{upper}} \\ 0 & \text{otherwise} \end{cases}$
- Veto mask: $\mathbb{1}_{\text{veto}} = \begin{cases} 1 & \text{if } \rho_t \geq \tau_{\text{veto}} \text{ for all } t \in T \\ 0 & \text{otherwise (any catastrophic token)} \end{cases}$

#### Properties

- **No IS Weights**: Uses pure rejection sampling
- **Extremely Sensitive**: Requires thresholds very close to 1.0 (e.g., 1.001)
- **Double Safety**: Geometric mean + per-token veto
- **High Rejection Rate**: Geometric mean amplifies deviations

**Why tight thresholds?**
A threshold of 1.001 is extremely tight. Geometric mean is highly sensitive to outliers:
- Arithmetic IS: $\prod_{t=1}^{100} \rho_t = 1.01^{100} \approx 2.7$
- Geometric IS: $(1.01)^{100 \cdot 1/100} = 1.01$

A threshold of 1.001 means rejecting sequences with an average per-token deviation over 0.1%.

---

## Veto Mechanism (Catastrophic Outlier Protection)

This is an independent safety layer that can be applied to all methods.

**Configuration:**
```python
RolloutCorrectionConfig(..., rollout_token_veto_threshold=1e-4)
```

**Veto Condition:**

$$
\text{Reject entire sequence if } \exists t \in T \text{ such that } \rho_t < \tau_{\text{veto}}
$$

This prevents catastrophic updates from tokens that $\pi_{\text{old}}$ assigned near-zero probability but $\pi_{\text{rollout}}$ sampled.

**Purpose:**
- Prevents catastrophic updates from sequences with extremely low token probabilities
- Independent of IS/RS settings (applied after IS/RS)
- Typical values: $10^{-4}$ to $10^{-6}$

**Implementation:** Line 620-640 in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L620-L640)

---

## Off-Policy Diagnostic Metrics

To monitor the severity of the off-policy gap (Drift 1: rollout $\to$ old/current), `verl` computes the following diagnostic metrics. These metrics help you determine whether off-policy correction is needed and which method to use.

**Note on notation:** The diagnostic metrics compare the **training policy** against $\pi_{\text{rollout}}$ (the behavior policy). The training policy depends on the mode:
- **Standard mode (no bypass)**: Training policy = $\pi_{\text{old}}$ (computed at start of epoch via `actor.compute_log_prob()`)
- **Bypass mode**: Training policy = $\pi_{\theta}$ (current policy parameters)

The IS ratio $\rho_t = \frac{\pi_{\text{train}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ measures the distribution shift from rollout to the training policy, where $\pi_{\text{train}}$ is whichever policy is being used as the training reference.

### KL Divergence

**Direct KL estimator:**

$$
\text{KL}(\pi_{\text{rollout}} \| \pi_{\text{train}}) = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \log \pi_{\text{rollout}}(a_t|s_t) - \log \pi_{\text{train}}(a_t|s_t) \right]
$$

**K3 KL estimator (more stable for small KL):**

$$
\text{KL}_{\text{K3}} = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t - \log \rho_t - 1 \right]
$$

where $\rho_t = \frac{\pi_{\text{train}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$.

### Perplexity

**Training policy:**

$$
\text{PPL}_{\text{train}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \pi_{\text{train}}(a_t|s_t) \right)
$$

**Rollout policy:**

$$
\text{PPL}_{\text{rollout}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \pi_{\text{rollout}}(a_t|s_t) \right)
$$

**PPL ratio (inverse of geometric IS weight):**

$$
\text{PPL}_{\text{ratio}} = \frac{\text{PPL}_{\text{train}}}{\text{PPL}_{\text{rollout}}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{-1/|T|}
$$

Equivalently: $\text{PPL}_{\text{ratio}} = \frac{1}{\text{geometric mean of } \rho_t}$

**Interpretation:** Higher values indicate the training policy assigns lower probability (higher perplexity) than the rollout policy, signaling a distribution shift.

### Chi-squared ($\chi^2$) Divergence

As established in prior work, this is the correct metric for diagnosing the potential for variance explosion.

**Token-level χ² divergence:**

$$
\chi^2_{\text{token}}(\pi_{\text{train}} \| \pi_{\text{rollout}}) = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t^2 \right] - 1
$$

where $\rho_t = \frac{\pi_{\text{train}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$.

**Sequence-level χ² divergence:**

$$
\chi^2_{\text{seq}}(\pi_{\text{train}} \| \pi_{\text{rollout}}) = \mathbb{E}_{\text{seq} \sim \pi_{\text{rollout}}} \left[ \left(\prod_{t \in T} \rho_t\right)^2 \right] - 1 = \mathbb{E}_{\text{seq}} \left[ \exp\left(2 \sum_{t \in T} \log \rho_t\right) \right] - 1
$$

**Interpretation:**
- $\chi^2 = 0$: Policies are identical
- $\chi^2 > 0$: Measures variance of IS weights (higher = more off-policy)
- Used to diagnose off-policy severity

**Implementation:** `compute_offpolicy_metrics()` in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L670-L776)

---

## Summary Table

| Method | IS Weight | RS | PPO Clip | Bypass | $\pi_{\text{old}}$ | IS Correction | Speed |
|--------|-----------|----|---------|---------|--------------------|---------------|-------|
| `token_is` | Per-token | ❌ | ✅ | ❌ | Computed | ✅ (rollout→old) | Standard |
| `seq_is` | Sequence | ❌ | ✅ | ❌ | Computed | ✅ (rollout→old) | Standard |
| `seq_is_rs` | Sequence | Sequence | ✅ | ❌ | Computed | ✅ (rollout→old) | Standard |
| `geo_rs` | ❌ | Geometric | ✅ | ❌ | Computed | ❌ (rejection only) | Standard |
| `ppo_is_bypass` | ~~Per-token~~ | ❌ | ✅ (vs rollout) | ✅ | **= Rollout** | ❌ (w=1.0) | **Faster** |
| `pure_is` | Sequence | ❌ | ❌ | ✅ | N/A | ✅ (direct θ→rollout) | **Faster** |

### Table Key

- ✅ = Enabled/Applied | ❌ = Disabled | ~~Strikethrough~~ = Degenerates to identity
- **Bypass**: Whether `bypass_old_logprob_for_rollout=True` (skips `actor.compute_log_prob()`)
- **$\pi_{\text{old}}$**: Source of old policy
  - **Computed**: Via `actor.compute_log_prob()` at start of training epoch
  - **= Rollout**: Reuses `rollout_log_prob` as `old_log_prob` → IS weight becomes 1.0
  - **N/A**: No separate old policy (pure_is compares current directly to rollout)
- **IS Correction**: How IS weights correct for distribution shift
  - **rollout→old**: Corrects drift from rollout to old policy (computed at epoch start)
  - **direct θ→rollout**: Corrects drift from current parameters to rollout (on-the-fly)
  - **w=1.0**: No actual correction (bypass makes old = rollout)
- **†Unbiased**: Only when threshold $\tau = \infty$; biased when truncated
- **Speed**: Relative computational cost. "Faster" methods skip the `actor.compute_log_prob()` call

### Method Comparison by Policy Count

| Method | Corrects Rollout→Current? | Mechanism | Total Policies Tracked |
|--------|--------------------------|-----------|----------------------|
| `token_is`, `seq_is` | ✅ Two-stage | IS weights (rollout→old) + PPO clip (old→current) | 3 policies |
| `ppo_is_bypass` | ✅ Single-stage | PPO clip (rollout→current directly) | 2 policies (old=rollout) |
| `pure_is` | ✅ Single-stage | IS weights (rollout→current directly) | 2 policies (no old) |
| Standard PPO | N/A | PPO clip (old→current only) | 2 policies (no rollout) |

**Trade-offs:**
- **3 policies (standard mode)**: Maximum observability - can separately measure Drift 1 (rollout→old) and Drift 2 (old→current). Cost: slower due to extra `old_log_prob` computation
- **2 policies (bypass mode)**: **Recommended for most users** - accepts that rollout ≈ old (valid assumption when rollout uses recent checkpoint). Benefit: faster, simpler
- **2 policies (pure_is)**: High variance approach - no PPO clipping safety net. Use when you need unbiased gradient estimates

**Key insight**: Computing `old_log_prob` separately is **only necessary** if you want to:
1. Measure the magnitude of Drift 1 (rollout→old off-policy gap) via IS weight statistics
2. Apply IS correction to Drift 1 while keeping PPO clipping for Drift 2

If `rollout_log_prob` is available and you trust your rollout policy is close to your training policy, **bypass mode is perfectly valid and often preferred**.

---

## Implementation References

- **[Rollout Correction Usage Guide](rollout_corr.md)** - Comprehensive guide for configuration, usage, metrics, and troubleshooting
- **Config:** [verl/trainer/config/algorithm.py](../../verl/trainer/config/algorithm.py)
- **IS/RS Helper:** [verl/trainer/ppo/rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py)
- **PPO Loss:** [verl/trainer/ppo/core_algos.py](../../verl/trainer/ppo/core_algos.py)
- **Tests:** [tests/trainer/ppo/test_rollout_corr.py](../../tests/trainer/ppo/test_rollout_corr.py)

---

## Reference

For more details, see:
- **Paper:** "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
  - Blog post: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
- **Off-Policy RL Theory:** https://fengyao.notion.site/off-policy-rl
