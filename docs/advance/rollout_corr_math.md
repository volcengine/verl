# Mathematical Formulations of Rollout Correction Methods in `verl`

**Author:** [Yingru Li](https://richardli.xyz)
**Last updated:** 2025-11-04

---

## Abstract

This document provides the definitive mathematical formulations for rollout correction methods in `verl`, following the natural progression from **REINFORCE** to **PPO** to **Decoupled PPO**.

The `verl` library implements **decoupled PPO** ([Hilton et al., 2021](https://arxiv.org/abs/2110.00641)) via a three-policy framework that separates the behavior policy (for off-policy correction) from the proximal policy (for trust region control). This enables off-policy RL with PPO-style stability, supporting scenarios where data collection differs from the training distribution.

**Applicable scenarios include:**
- **Policy mismatch**: Different precision (FP8 vs FP16 vs BF16 vs FP32), different backends (vLLM vs SGLang vs FSDP vs Megatron)
- **Temporal lag**: Model staleness, asynchronous rollout workers
- **Replay buffers**: Training on historical trajectories from earlier policy versions
- **Off-policy algorithms**: Behavioral cloning, DAPO, expert demonstrations
- **Data filtering**: Reweighting, preference learning, curriculum learning

---

## Table of Contents

1. [Theoretical Foundation: From REINFORCE to Decoupled PPO](#1-theoretical-foundation-from-reinforce-to-decoupled-ppo)
2. [Implementation in verl: The Three-Policy Framework](#2-implementation-in-verl-the-three-policy-framework)
3. [Method Variants: Different Algorithmic Choices](#3-method-variants-different-algorithmic-choices)
4. [Safety Mechanisms and Rejection Sampling](#4-safety-mechanisms-and-rejection-sampling)
5. [Off-Policy Diagnostic Metrics](#5-off-policy-diagnostic-metrics)
6. [Summary and Decision Guide](#6-summary-and-decision-guide)
7. [Implementation References](#7-implementation-references)

---

## 1. Theoretical Foundation: From REINFORCE to Decoupled PPO

This section establishes the theoretical progression that `verl` implements.

### 1.1 REINFORCE: Policy Gradient Baseline

The REINFORCE algorithm ([Williams, 1992](https://doi.org/10.1007/BF00992696)) is the foundation of policy gradient methods. For data sampled from a behavior policy $\mu$, the policy gradient with importance sampling is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \mu} \left[ \frac{\pi_\theta(a|s)}{\mu(a|s)} \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]
$$

**Key properties:**
- **Off-policy capable**: Can learn from any behavior policy via importance sampling
- **High variance**: Gradient estimates can be unstable
- **No trust region**: Large policy updates can destabilize training

**Implementation in verl:** The `pure_is` method implements REINFORCE with truncated importance sampling.

### 1.2 PPO: Adding Trust Region Control

Proximal Policy Optimization ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) adds a clipped surrogate objective to stabilize training:

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ and $\epsilon$ is the clip range (typically 0.2).

**Key properties:**
- **Two policies**: $\pi_{\text{old}}$ (reference/anchor) and $\pi_\theta$ (being updated)
- **Trust region via clipping**: Prevents destructive large policy updates
- **On-policy**: Original PPO assumes data is collected from $\pi_{\text{old}}$
- **Stable training**: Significantly more stable than vanilla REINFORCE

**Limitation:** Standard PPO assumes on-policy data (data collected from $\pi_{\text{old}}$). When data comes from a different policy, off-policy correction is needed.

### 1.3 Decoupled PPO: Enabling Off-Policy PPO

Decoupled PPO ([Hilton et al., 2021](https://arxiv.org/abs/2110.00641)) extends PPO to handle off-policy data by **separating two roles**:
1. **Behavior policy** $\mu$: The policy that collected the data (for importance sampling correction)
2. **Proximal policy** $\pi_{\text{ref}}$: The anchor policy for PPO clipping (for trust region control)

This leads to a **three-policy formulation**:

$$
L_{\text{DecoupledPPO}}(\theta) = -\mathbb{E}_{(s,a) \sim \mu} \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $w_t = \frac{\pi_{\text{ref}}(a_t|s_t)}{\mu(a_t|s_t)}$: Importance sampling weight (corrects for off-policy data)
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$: PPO ratio (trust region control)

**Key insight:** By decoupling $\mu$ from $\pi_{\text{ref}}$, we can:
- Use **any behavior policy** $\mu$ for data collection (off-policy)
- Maintain **PPO-style stability** via clipping against $\pi_{\text{ref}}$
- Achieve **batch size invariance** (can aggregate data across multiple rollout workers)
- Utilize **stale data** (replay buffers, asynchronous workers)

**This is the algorithm that `verl` implements via its three-policy framework.**

---

## 2. Implementation in verl: The Three-Policy Framework

The `verl` library implements decoupled PPO using three distinct policies, each serving a specific role.

### 2.1 Policy Roles and Notation

**$\pi_{\text{rollout}}$ (Behavior Policy $\mu$)**
The policy used for data collection. This is the behavior distribution $\mu$ from theory.

- **When created**: During rollout/data collection phase
- **Purpose**: Generate trajectories for training
- **Common sources**:
  - Policy mismatch: Same weights, different implementation (precision, backend)
  - Temporal lag: Stale checkpoint from async workers
  - Replay buffer: Historical data from earlier iterations
  - Off-policy algorithms: Expert demonstrations, auxiliary policies (DAPO)
  - Data filtering: Reweighted or filtered data
- **Fixed**: Frozen during training on a batch

**$\pi_{\text{old}}$ (Proximal Policy $\pi_{\text{ref}}$)**
The reference policy for PPO clipping. This is the "proximal policy" from decoupled PPO theory.

- **When created**:
  - **Standard mode**: Computed at start of training epoch via `actor.compute_log_prob()`
  - **Bypass mode**: Set equal to $\pi_{\text{rollout}}$ (skips separate computation)
- **Purpose**:
  - Anchor point for PPO clipping (trust region control)
  - Enables decoupled PPO when separate from $\pi_{\text{rollout}}$
- **Fixed**: Frozen during all PPO update epochs on the same batch

**$\pi_{\theta}$ (Current Policy)**
The policy being actively optimized during training.

- **Updated**: Every gradient step
- **Purpose**: The policy we're improving

### 2.2 Operating Modes

The three-policy framework can operate in two modes:

**Standard Mode (Three Policies)**
- Computes $\pi_{\text{old}}$ separately at the start of each training epoch
- **Algorithm**: Full decoupled PPO with three policies
- **Use when**: You need maximum correction accuracy and can afford the computational cost
- **Benefits**: Separately corrects Drift 1 (rollout→old) and Drift 2 (old→current)

**Bypass Mode (Two Policies)**
- Sets $\pi_{\text{old}} = \pi_{\text{rollout}}$ (skips separate computation)
- **Algorithm**: Degenerates to either standard PPO or off-policy REINFORCE
- **Use when**: $\pi_{\text{rollout}} \approx \pi_{\text{old}}$ (e.g., recent checkpoint, minor implementation differences)
- **Benefits**: Faster (skips `actor.compute_log_prob()` call)

### 2.3 Two Distribution Shifts

The three-policy framework handles two types of distribution drift:

**Drift 1: $\pi_{\text{rollout}} \to \pi_{\text{old}}$ (Off-Policy Gap)**

This is the distribution shift between the data collection policy and the training reference policy.

- **Nature**: Ranges from negligible (same checkpoint, minor differences) to severe (replay buffers, expert data)
- **Correction**: Importance sampling weight $w_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$
- **Optional**: Can be ignored (bypass mode) when negligible

**Drift 2: $\pi_{\text{old}} \to \pi_{\theta}$ (Policy Update Drift)**

This is the drift from policy parameter updates during training.

- **Nature**: Occurs as $\pi_\theta$ is updated via gradient descent
- **Correction**: PPO clipping on ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- **Universal**: Applies to both on-policy and off-policy training

### 2.4 Notation Summary

- $\pi_{\text{rollout}}$: Behavior policy (data collection)
- $\pi_{\text{old}}$: Proximal policy (PPO anchor)
- $\pi_{\theta}$: Current policy (being updated)
- $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$: Per-token IS ratio (corrects Drift 1)
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: PPO ratio (corrects Drift 2)
- $A_t$: Advantage at token $t$
- $T$: Set of valid tokens in a sequence
- $\tau$: Upper threshold for IS weights (e.g., 2.0)
- $\tau_{\text{lower}}$: Lower threshold for IS weights (typically $1/\tau$)
- $\epsilon$: PPO clip range (typically 0.2)

---

## 3. Method Variants: Different Algorithmic Choices

This section describes the different algorithmic variants available in `verl`, organized by their theoretical foundation.

### 3.1 Off-Policy REINFORCE Methods

These methods implement REINFORCE with importance sampling, without PPO clipping.

#### 3.1.1 Pure IS (pure_is)

**Theory:** Off-policy REINFORCE with sequence-level truncated importance sampling.

**Configuration:**
```python
RolloutCorrectionConfig.pure_is(threshold=2.0)
```

**Loss Function:**

$$
L_{\text{PureIS}}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ w_{\text{seq}}(\theta) \cdot \sum_{t \in T} \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

where:
- Sequence-level IS weight: $w_{\text{seq}}(\theta) = \min\left( \prod_{t \in T} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, \tau \right)$
- IS weight is **detached from gradient** (treated as constant)
- Direct comparison: $\pi_\theta$ to $\pi_{\text{rollout}}$

**Effective gradient:**

$$
\nabla_\theta L_{\text{PureIS}} = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ \text{stopgrad}(w_{\text{seq}}(\theta)) \cdot \sum_{t \in T} \nabla_\theta \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

**Properties:**
- **Algorithm**: Off-policy REINFORCE + IS
- **Policies**: Two ($\pi_{\text{rollout}}$, $\pi_\theta$)
- **No PPO clipping**: Pure policy gradient
- **Always uses bypass mode**: No $\pi_{\text{old}}$ computation
- **Fast**: Single forward pass for IS weights
- **Use when**: You want pure policy gradient with off-policy correction

**Implementation:** `compute_policy_loss_with_rollout_correction()` in [core_algos.py](verl/trainer/ppo/core_algos.py#L1537-L1681)

---

### 3.2 Standard PPO Methods

These methods apply PPO clipping without importance sampling for off-policy correction.

#### 3.2.1 Standard PPO (On-Policy)

**Theory:** Original PPO, assumes data is collected from $\pi_{\text{old}}$ (on-policy).

**Loss Function:**

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$.

**Properties:**
- **Algorithm**: Standard PPO
- **Policies**: Two ($\pi_{\text{old}}$, $\pi_\theta$)
- **Ignores $\pi_{\text{rollout}}$**: Assumes on-policy data
- **Use when**: Data is truly on-policy (or off-policy gap is negligible)

**Implementation:** `compute_policy_loss()` in [core_algos.py](verl/trainer/ppo/core_algos.py#L812-L884)

#### 3.2.2 PPO Bypass (ppo_is_bypass)

**Theory:** Original PPO applied to off-policy data by using $\pi_{\text{rollout}}$ as the PPO anchor.

**Configuration:**
```python
RolloutCorrectionConfig.ppo_is_bypass(threshold=2.0)
```

**Key insight:** When `bypass_old_logprob_for_rollout=True`, we set $\pi_{\text{old}} = \pi_{\text{rollout}}$:
- IS weight: $w_t = \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}} = 1$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}}{\pi_{\text{old}}} = \frac{\pi_{\theta}}{\pi_{\text{rollout}}}$

**Loss Function:**

$$
L_{\text{PPO-Bypass}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (clips against rollout policy).

**Properties:**
- **Algorithm**: Standard PPO (two policies)
- **Policies**: Two ($\pi_{\text{rollout}}$, $\pi_\theta$)
- **No IS correction**: Assumes $\pi_{\text{rollout}} \approx \pi_{\text{old}}$
- **PPO clips against rollout**: Trust region relative to data collection policy
- **Fast**: Skips `actor.compute_log_prob()` call
- **Use when**: Rollout policy is close to what old policy would be (e.g., recent checkpoint)

---

### 3.3 Decoupled PPO Methods

These methods implement full decoupled PPO with three policies, combining importance sampling (for Drift 1) with PPO clipping (for Drift 2).

#### 3.3.1 Token-Level IS (token_is)

**Theory:** Decoupled PPO with per-token truncated importance sampling.

**Configuration:**
```python
RolloutCorrectionConfig.token_is(threshold=2.0)
```

**Loss Function:**

$$
L_{\text{PPO+TIS}}(\theta) = -\mathbb{E}_t \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- Per-token IS weight: $w_t = \min(\rho_t, \tau) = \min\left(\frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, \tau\right)$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- $\pi_{\text{old}}$ is computed at **start of training epoch**

**Properties:**
- **Algorithm**: Decoupled PPO
- **Policies**: Three ($\pi_{\text{rollout}}$, $\pi_{\text{old}}$, $\pi_\theta$) in standard mode
- **Double correction**: IS weights correct Drift 1, PPO clips correct Drift 2
- **Recommended default**: Best for most off-policy scenarios with moderate drift
- **Per-token truncation**: Stable IS weight computation

**Implementation:**
- IS weights: `compute_rollout_correction_weights()` in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L325-L402)
- Loss: `compute_policy_loss()` in [core_algos.py](verl/trainer/ppo/core_algos.py#L812-L884)

#### 3.3.2 Sequence-Level IS (seq_is)

**Theory:** Decoupled PPO with sequence-level importance sampling.

**Configuration:**
```python
RolloutCorrectionConfig.seq_is(threshold=2.0)
```

**Loss Function:**

$$
L_{\text{PPO+SeqIS}}(\theta) = -\mathbb{E}_t \left[ w_{\text{seq}} \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- Sequence-level IS weight (broadcast to all tokens):
$$w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, \tau \right) = \min\left( \exp\left(\sum_{t \in T} \log \rho_t\right), \tau \right)$$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$

**Properties:**
- **Algorithm**: Decoupled PPO
- **Policies**: Three ($\pi_{\text{rollout}}$, $\pi_{\text{old}}$, $\pi_\theta$) in standard mode
- **Sequence-level IS**: Uses product of all token ratios
- **More aggressive**: Higher variance than token-level, but potentially more accurate

#### 3.3.3 Mixed IS + Rejection Sampling (seq_is_rs / seq_mis)

**Theory:** Decoupled PPO combining sequence-level IS weighting with rejection sampling.

**Configuration:**
```python
RolloutCorrectionConfig.seq_is_rs(
    is_threshold=2.0,
    rs_threshold=2.0,
    rs_threshold_lower=None,  # defaults to 1/rs_threshold
)
```

**Loss Function:**

$$
L_{\text{PPO+MIS}}(\theta) = -\mathbb{E}_{t \mid \text{seq} \in \mathcal{A}} \left[ w_{\text{seq}} \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- IS weight: $w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, \tau_{\text{IS}} \right)$
- Acceptance set: $\mathcal{A} = \{ \text{seq} : \tau_{\text{lower}} \leq \prod_{t \in T} \rho_t \leq \tau_{\text{upper}} \}$

**Properties:**
- **Algorithm**: Decoupled PPO + rejection sampling
- **Double mechanism**: IS reweighting + rejection filtering
- **Lower effective sample size**: Rejects outlier sequences
- **Use when**: Severe off-policy distribution shift with outliers

**Implementation:** `compute_rollout_rejection_mask()` in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L80-L188)

---

## 4. Safety Mechanisms and Rejection Sampling

### 4.1 Geometric Rejection Sampling (geo_rs)

**Theory:** Pure rejection sampling based on geometric mean of IS ratios.

**Configuration:**
```python
RolloutCorrectionConfig.geo_rs(
    rs_threshold=1.001,  # Very tight threshold
    rs_threshold_lower=None,
    veto_threshold=1e-4,
)
```

**Loss Function:**

$$
L_{\text{GeoRS}}(\theta) = -\mathbb{E}_{t \mid \text{seq} \in \mathcal{A}_{\text{geo}} \cap \mathcal{A}_{\text{veto}}} \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- Geometric mean: $\rho_{\text{geo}} = \exp\left( \frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{1/|T|}$
- Geometric acceptance: $\mathcal{A}_{\text{geo}} = \{ \text{seq} : \tau_{\text{lower}} \leq \rho_{\text{geo}} \leq \tau_{\text{upper}} \}$
- Veto acceptance: $\mathcal{A}_{\text{veto}} = \{ \text{seq} : \rho_t \geq \tau_{\text{veto}} \text{ for all } t \in T \}$

**Why tight thresholds?**
Geometric mean is extremely sensitive. For 100 tokens with $\rho_t = 1.01$ each:
- Arithmetic product: $\prod_{t=1}^{100} \rho_t = 1.01^{100} \approx 2.7$
- Geometric mean: $(1.01)^{1} = 1.01$

A threshold of 1.001 means rejecting sequences with average per-token deviation > 0.1%.

**Properties:**
- **No IS weights**: Pure rejection
- **Extremely selective**: Requires near-perfect policy match
- **High rejection rate**: Suitable for very slight distribution shifts only

### 4.2 Veto Mechanism

An independent safety layer that rejects sequences with catastrophically low token probabilities.

**Configuration:**
```python
RolloutCorrectionConfig(..., rollout_token_veto_threshold=1e-4)
```

**Veto condition:**

$$
\text{Reject entire sequence if } \exists t \in T \text{ such that } \rho_t < \tau_{\text{veto}}
$$

**Purpose:**
- Prevents catastrophic updates from tokens with near-zero probability under $\pi_{\text{old}}$
- Independent of IS/RS settings
- Typical values: $10^{-4}$ to $10^{-6}$

**Implementation:** [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L620-L640)

---

## 5. Off-Policy Diagnostic Metrics

These metrics help monitor the severity of off-policy drift and guide method selection.

**Note on notation:** Metrics use $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$. In bypass mode, $\pi_{\text{old}} = \pi_{\text{rollout}}$, so metrics measure rollout→current drift using $\rho_t = \frac{\pi_{\theta}}{\pi_{\text{rollout}}}$ instead.

### 5.1 KL Divergence

**Direct KL estimator:**

$$
\text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}}) = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \log \pi_{\text{rollout}}(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t) \right]
$$

**K3 KL estimator** (more stable for small KL):

$$
\text{KL}_{\text{K3}} = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t - \log \rho_t - 1 \right]
$$

where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$.

### 5.2 Perplexity

**Old policy perplexity:**

$$
\text{PPL}_{\text{old}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \pi_{\text{old}}(a_t|s_t) \right)
$$

**Rollout policy perplexity:**

$$
\text{PPL}_{\text{rollout}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \pi_{\text{rollout}}(a_t|s_t) \right)
$$

**PPL ratio** (inverse of geometric mean IS weight):

$$
\text{PPL}_{\text{ratio}} = \frac{\text{PPL}_{\text{old}}}{\text{PPL}_{\text{rollout}}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{-1/|T|}
$$

**Interpretation:** Values > 1 mean $\pi_{\text{old}}$ assigns lower probability than $\pi_{\text{rollout}}$ to the observed actions (distribution shift).

### 5.3 Chi-squared Divergence

Measures the second moment of the IS weight distribution.

**Token-level:**

$$
\chi^2_{\text{token}} = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t^2 \right] - 1
$$

**Sequence-level:**

$$
\chi^2_{\text{seq}} = \mathbb{E}_{\text{seq} \sim \pi_{\text{rollout}}} \left[ \left(\prod_{t \in T} \rho_t\right)^2 \right] - 1
$$

**Interpretation:**
- $\chi^2 = 0$: Policies are identical
- $\chi^2 > 0$: Higher values indicate more severe off-policy distribution shift

**Implementation:** `compute_offpolicy_metrics()` in [rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L670-L776)

---

## 6. Summary and Decision Guide

### 6.1 Method Summary Table

| Method | Theory | Policies | PPO Clip | IS Correction | Bypass | Speed |
|--------|--------|----------|----------|---------------|--------|-------|
| `pure_is` | Off-policy REINFORCE | 2 (rollout, θ) | ❌ | ✅ Seq-level | Always | **Fast** |
| Standard PPO | PPO | 2 (old, θ) | ✅ | ❌ | N/A | Standard |
| `ppo_is_bypass` | PPO | 2 (rollout, θ) | ✅ | ❌ | Always | **Fast** |
| `token_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Token-level | ❌ | Standard |
| `seq_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Seq-level | ❌ | Standard |
| `seq_is_rs` | Decoupled PPO + RS | 3 (rollout, old, θ) | ✅ | ✅ + Rejection | ❌ | Standard |
| `geo_rs` | PPO + Geo RS | 3 (rollout, old, θ) | ✅ | Rejection only | ❌ | Standard |

### 6.2 Decision Guide

**Choose your method based on:**

1. **Off-policy severity:**
   - **Negligible** (same checkpoint, minor differences): `ppo_is_bypass` or standard PPO
   - **Moderate** (async workers, slight staleness): `token_is` (recommended default)
   - **Severe** (replay buffers, old data): `seq_is` or `seq_is_rs`

2. **Algorithm preference:**
   - **Want decoupled PPO benefits** (batch size invariance, stale data): Use standard mode (`token_is`, `seq_is`)
   - **Want computational efficiency**: Use bypass mode (`ppo_is_bypass`)
   - **Want pure policy gradient**: Use `pure_is`

3. **Stability requirements:**
   - **Need PPO stability**: Any method except `pure_is`
   - **Accept higher variance for simplicity**: `pure_is`

### 6.3 When to Compute $\pi_{\text{old}}$ Separately

Computing `old_log_prob` separately (standard mode, no bypass) is **only necessary** if you want to:

1. **Implement full decoupled PPO** with three policies
2. **Separately measure and correct** Drift 1 (rollout→old) and Drift 2 (old→current)
3. **Enable batch size invariance** and better stale data utilization
4. **Monitor off-policy metrics** accurately

**Bypass mode is valid when:**
- $\pi_{\text{rollout}} \approx \pi_{\text{old}}$ (recent checkpoint, minor implementation differences)
- Computational efficiency is priority
- Off-policy gap is negligible

---

## 7. Implementation References

- **[Rollout Correction Usage Guide](rollout_corr.md)** - Practical configuration and troubleshooting
- **Config:** [verl/trainer/config/algorithm.py](../../verl/trainer/config/algorithm.py)
- **IS/RS Helper:** [verl/trainer/ppo/rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py)
- **PPO Loss:** [verl/trainer/ppo/core_algos.py](../../verl/trainer/ppo/core_algos.py)
- **Tests:** [tests/trainer/ppo/test_rollout_corr.py](../../tests/trainer/ppo/test_rollout_corr.py)

---

## References

- **Williams, R. J. (1992).** "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256. https://doi.org/10.1007/BF00992696
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347.* https://arxiv.org/abs/1707.06347
- **Hilton, J., Cobbe, K., & Schulman, J. (2021).** "Batch size-invariance for policy optimization." *arXiv preprint arXiv:2110.00641.* https://arxiv.org/abs/2110.00641
  - Introduced decoupled PPO: separating behavior policy (off-policy correction) from proximal policy (trust region)
- **Li, Y.** "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
  - Blog post: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
- **Off-Policy RL Theory:** https://fengyao.notion.site/off-policy-rl
