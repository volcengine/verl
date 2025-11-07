# Mathematical Formulations of Rollout Correction Methods in `verl`

**Author:** [Yingru Li](https://richardli.xyz)
**Last updated:** 2025-11-04

---

## Abstract

This document provides the definitive mathematical formulations for rollout correction methods in `verl`, following the natural progression from **REINFORCE** to **PPO** to **Decoupled PPO**.

Rollout correction provides a unified framework to handle **general off-policy problems** in RL training - any scenario where the data collection distribution differs from the training distribution.

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

The REINFORCE algorithm ([Williams, 1992](https://doi.org/10.1007/BF00992696)) is the foundation of policy gradient methods.

**Vanilla REINFORCE (On-Policy)**

For trajectories $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$ sampled from the current policy $\pi_\theta$, the policy gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $A_t$ is the advantage function at timestep $t$.

**Off-Policy REINFORCE**

When trajectories are sampled from a different behavior policy $\mu$, we apply importance sampling over the **joint trajectory distribution**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \mu} \left[ \frac{P_{\pi_\theta}(\tau)}{P_\mu(\tau)} \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where the trajectory-level importance weight is:

$$
\frac{P_{\pi_\theta}(\tau)}{P_\mu(\tau)} = \frac{p(s_0) \prod_{t=0}^T \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)}{p(s_0) \prod_{t=0}^T \mu(a_t|s_t) p(s_{t+1}|s_t, a_t)} = \prod_{t=0}^T \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)}
$$

The transition dynamics $p(s_{t+1}|s_t, a_t)$ and initial state $p(s_0)$ cancel out, leaving only the product of per-step action probability ratios.

**Key properties:**
- **Off-policy capable**: Can learn from any behavior policy via importance sampling
- **No trust region**: Policy updates not constrained

**Implementation in verl:** The `pure_is` method implements off-policy REINFORCE with truncated importance sampling.

### 1.2 PPO: Adding Trust Region Control

Proximal Policy Optimization ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) adds a clipped surrogate objective:

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_{(s,a) \sim \mu} \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)}$ and $\epsilon$ is the clip range (typically 0.2).

**Key properties:**
- **Two policies**: $\mu$ (reference for clipping) and $\pi_\theta$ (being updated)
- **Trust region via clipping**: Limits policy update magnitude via ratio $r_t(\theta) = \frac{\pi_\theta}{\mu}$

### 1.3 Decoupled PPO: Achieving Batch Size Invariance

Decoupled PPO ([Hilton et al., 2021](https://arxiv.org/abs/2110.00641)) solves PPO's batch size sensitivity by **decoupling two roles**:
1. **Proximal policy** $\pi_{\text{prox}}$: The anchor policy for PPO clipping (controls policy update size)
2. **Behavior policy** $\mu$: The policy that collected the data (for off-policy correction via importance sampling)

**The problem**: Standard PPO controls policy update size via the ratio $\frac{\pi_\theta}{\pi_{\text{old}}}$, where $\pi_{\text{old}}$ is assumed to be both the proximal policy *and* the behavior policy. This coupling makes the algorithm sensitive to batch size because aggregating data from multiple workers or using replay buffers changes the effective behavior policy.

**The solution**: Decouple these two roles, leading to a **three-policy formulation**:

$$
L_{\text{DecoupledPPO}}(\theta) = -\mathbb{E}_{(s,a) \sim \mu} \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $w_t = \frac{\pi_{\text{prox}}(a_t|s_t)}{\mu(a_t|s_t)}$: Importance sampling weight (corrects for behavior policy $\mu$)
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{prox}}(a_t|s_t)}$: PPO ratio (controls policy update size against proximal policy $\pi_{\text{prox}}$)

**Key properties**: By decoupling:
- **Batch size invariance**: Policy update control (via $\pi_{\text{prox}}$) is independent of data aggregation
- **Flexible behavior policy**: Any $\mu$ can be used (different workers, replay buffers, or stale checkpoints)
- **Stale data utilization**: Older trajectories can be corrected via importance sampling
- **Clipping preserved**: Clipping against $\pi_{\text{prox}}$ limits update magnitude

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

**$\pi_{\text{old}}$ (Proximal Policy $\pi_{\text{prox}}$)**
The reference policy for PPO clipping. This is the "proximal policy" from decoupled PPO theory.

- **When created**:
  - **Decoupled mode**: Computed at start of training epoch via `actor.compute_log_prob()`
  - **Bypass mode**: Set equal to $\pi_{\text{rollout}}$ (skips separate computation)
- **Purpose**:
  - Anchor point for PPO clipping (controls policy update size)
  - When separate from $\pi_{\text{rollout}}$: Enables batch size invariance and efficient use of stale data
- **Fixed**: Frozen during all PPO update epochs on the same batch

**$\pi_{\theta}$ (Current Policy)**
The policy being actively optimized during training.

- **Updated**: Every gradient step
- **Purpose**: The policy we're improving

### 2.2 Operating Modes

The three-policy framework can operate in two modes:

**Decoupled Mode (Three Policies)**
- Computes $\pi_{\text{old}}$ separately at the start of each training epoch
- **Algorithm**: Full decoupled PPO with three policies (mathematically correct)
- **Properties**: Achieves batch size invariance; separately corrects Drift 1 (rollout→old) and Drift 2 (old→current)

**Bypass Mode (Two Policies)**
- Sets $\pi_{\text{old}} = \pi_{\text{rollout}}$ (skips separate computation)
- **Algorithm**: Uses $\pi_{\text{rollout}}$ as both behavior policy and proximal policy (mathematically correct)
- **Key difference**: Proximal policy equals behavior policy, so no IS correction needed between them
- **Properties**: Faster (skips `actor.compute_log_prob()` call); does not achieve batch size invariance

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
- $C_{\text{IS}}$: Upper threshold for IS weights (e.g., 2.0)
- $C_{\text{RS-upper}}$: Upper threshold for RS mask (e.g., 2.0)
- $C_{\text{RS-lower}}$: Lower threshold for RS mask (typically $1/C_{\text{RS-upper}}$)
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
- Sequence-level IS weight: $w_{\text{seq}}(\theta) = \min\left( \prod_{t \in T} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, C_{\text{IS}} \right)$
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

**Implementation:** `compute_policy_loss_with_rollout_correction()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L1537-L1681)

---

### 3.2 Two-Policy PPO Methods

These methods use two policies without importance sampling between behavior and proximal policies.

#### 3.2.1 Incorrect LLM-RL Implementation (PPO Without Rollout Correction)

**Theory:** Naive LLM-RL implementation that incorrectly applies PPO by ignoring the actual rollout policy and assuming $\mu = \pi_{\text{old}}$.

**Note:** This incorrect implementation pattern was identified in [Liu, Li, et al. (2025)](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) as a key cause of training instability in LLM-RL systems, motivating the development of this rollout correction framework.

**Loss Function:**

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$.

**Properties:**
- **Algorithm**: Common but incorrect LLM-RL implementation (mathematically wrong when $\pi_{\text{rollout}} \neq \pi_{\text{old}}$)
- **Policies**: Two ($\pi_{\text{old}}$, $\pi_\theta$)
- **Ignores $\pi_{\text{rollout}}$**: Uses $\pi_{\text{old}}$ as behavior policy instead of actual $\pi_{\text{rollout}}$
- **Policy mismatch**: This is the typical case in LLM-RL - rollout uses different precision/backend/checkpoint than training, causing $\pi_{\text{rollout}} \neq \pi_{\text{old}}$ even with same model weights
- **Not PPO's fault**: PPO itself is correct; the issue is the incorrect assumption that $\pi_{\text{old}} = \pi_{\text{rollout}}$ in LLM-RL implementations

**Implementation:** `compute_policy_loss()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L812-L884)

#### 3.2.2 PPO Bypass (ppo_is_bypass)

**Theory:** Original PPO applied to off-policy data by using $\pi_{\text{rollout}}$ as the PPO anchor.

**Configuration:**
```python
RolloutCorrectionConfig.ppo_is_bypass(threshold=2.0)
```

**Implementation:** When `bypass_old_logprob_for_rollout=True`, we set $\pi_{\text{old}} = \pi_{\text{rollout}}$:
- IS weight: $w_t = \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}} = 1$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}}{\pi_{\text{old}}} = \frac{\pi_{\theta}}{\pi_{\text{rollout}}}$

**Loss Function:**

$$
L_{\text{PPO-Bypass}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (clips against rollout policy).

**Properties:**
- **Algorithm**: PPO with $\pi_{\text{rollout}}$ as proximal policy (two policies)
- **Policies**: Two ($\pi_{\text{rollout}}$, $\pi_\theta$)
- **No IS correction needed**: Uses actual behavior policy $\pi_{\text{rollout}}$ as proximal policy (mathematically correct)
- **PPO clips against rollout**: Trust region relative to data collection policy
- **Fast**: Skips `actor.compute_log_prob()` call

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
- Per-token IS weight: $w_t = \min(\rho_t, C_{\text{IS}}) = \min\left(\frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, C_{\text{IS}} \right)$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- $\pi_{\text{old}}$ is computed at **start of training epoch**

**Properties:**
- **Algorithm**: Decoupled PPO
- **Policies**: Three ($\pi_{\text{rollout}}$, $\pi_{\text{old}}$, $\pi_\theta$) in decoupled mode
- **Double correction**: IS weights correct Drift 1, PPO clips correct Drift 2
- **Per-token truncation**: Stable IS weight computation

**Implementation:**
- IS weights: `compute_rollout_correction_weights()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L325-L402)
- Loss: `compute_policy_loss()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L812-L884)

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
$$w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, C_{\text{IS}} \right) = \min\left( \exp\left(\sum_{t \in T} \log \rho_t\right), C_{\text{IS}} \right)$$
- PPO ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$

**Properties:**
- **Algorithm**: Decoupled PPO
- **Policies**: Three ($\pi_{\text{rollout}}$, $\pi_{\text{old}}$, $\pi_\theta$) in decoupled mode
- **Sequence-level IS**: Uses product of all token ratios

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
- IS weight: $w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, C_{\text{IS}} \right)$
- Acceptance set: $\mathcal{A} = \{ \text{seq} : C_{\text{RS-lower}} \leq \prod_{t \in T} \rho_t \leq C_{\text{RS-upper}} \}$

**Properties:**
- **Algorithm**: Decoupled PPO + rejection sampling
- **Double mechanism**: IS reweighting + rejection filtering
- **Lower effective sample size**: Rejects outlier sequences

**Implementation:** `compute_rollout_rejection_mask()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L80-L188)

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
- Geometric acceptance: $\mathcal{A}_{\text{geo}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \rho_{\text{geo}} \leq C_{\text{RS-upper}} \}$
- Veto acceptance: $\mathcal{A}_{\text{veto}} = \{ \text{seq} : \rho_t \geq C_{\text{veto}} \text{ for all } t \in T \}$

**Why tight thresholds?**
Geometric mean is extremely sensitive. For 100 tokens with $\rho_t = 1.01$ each:
- Arithmetic product: $\prod_{t=1}^{100} \rho_t = 1.01^{100} \approx 2.7$
- Geometric mean: $(1.01)^{1} = 1.01$

A threshold of 1.001 means rejecting sequences with average per-token deviation > 0.1%.

**Properties:**
- **No IS weights**: Pure rejection
- **Extremely selective**: Requires near-perfect policy match
- **High rejection rate**: Only suitable for very slight distribution shifts

### 4.2 Veto Mechanism

An independent safety layer that rejects sequences with catastrophically low token probabilities.

**Configuration:**
```python
RolloutCorrectionConfig(..., rollout_token_veto_threshold=1e-4)
```

**Veto condition:**

$$
\text{Reject entire sequence if } \exists t \in T \text{ such that } \rho_t < C_{\text{veto}}
$$

**Purpose:**
- Prevents catastrophic updates from tokens with near-zero probability under $\pi_{\text{old}}$
- Independent of IS/RS settings
- Typical values: $10^{-4}$ to $10^{-6}$

**Implementation:** [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L620-L640)

---

## 5. Off-Policy Diagnostic Metrics

These metrics quantify the severity of off-policy drift.

**Note on notation:** Metrics use $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$. In bypass mode, $\pi_{\text{old}} = \pi_{\text{rollout}}$, so metrics measure rollout→current drift using $\rho_t = \frac{\pi_{\theta}}{\pi_{\text{rollout}}}$ instead.

### 5.1 KL Divergence

**Direct KL estimator:**

$$
\text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}}) = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \log \pi_{\text{rollout}}(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t) \right]
$$

**K3 KL estimator** (alternative formulation):

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

**Implementation:** `compute_offpolicy_metrics()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L670-L776)

---

## 6. Summary and Decision Guide

### 6.1 Method Summary Table

| Method | Theory | Policies | PPO Clip | IS Correction | Correctness | Speed |
|--------|--------|----------|----------|---------------|-------------|-------|
| `pure_is` | Off-policy REINFORCE | 2 (rollout, θ) | ❌ | ✅ Seq-level | ✅ Correct | **Fast** |
| Naive LLM-RL | Incorrect PPO usage | 2 (old, θ) | ✅ | ❌ | ⚠️ Incorrect | Standard |
| `ppo_is_bypass` | PPO (rollout as prox) | 2 (rollout, θ) | ✅ | ❌ | ✅ Correct | **Fast** |
| `token_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Token-level | ✅ Correct | Standard |
| `seq_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Seq-level | ✅ Correct | Standard |
| `seq_is_rs` | Decoupled PPO + RS | 3 (rollout, old, θ) | ✅ | ✅ + Rejection | ✅ Correct | Standard |
| `geo_rs` | Decoupled PPO + Geo RS | 3 (rollout, old, θ) | ✅ | Rejection only | ✅ Correct | Standard |

### 6.2 Method Characteristics by Scenario

**Off-policy severity:**
- **Negligible** (same checkpoint, minor differences): `ppo_is_bypass` uses $\pi_{\text{rollout}}$ as proximal policy (mathematically correct); naive LLM-RL implementations use $\pi_{\text{old}}$ instead of $\pi_{\text{rollout}}$ (mathematically incorrect when $\pi_{\text{rollout}} \neq \pi_{\text{old}}$)
- **Moderate** (async workers, slight staleness): `token_is` provides per-token IS correction with separate proximal policy
- **Severe** (replay buffers, old data): `seq_is` and `seq_is_rs` provide sequence-level IS correction with optional rejection sampling

**Algorithm properties:**
- **Batch size invariance**: Decoupled mode with three policies (`token_is`, `seq_is`) achieves batch size invariance
- **Computational efficiency**: Bypass mode (`ppo_is_bypass`) skips `old_log_prob` computation
- **Pure policy gradient**: `pure_is` implements off-policy REINFORCE without PPO clipping

### 6.3 Decoupled Mode vs Bypass Mode

**Decoupled mode** (computes `old_log_prob` separately):
- Implements full decoupled PPO with three policies (mathematically correct)
- Separately measures and corrects Drift 1 (rollout→old) and Drift 2 (old→current)
- Achieves batch size invariance and efficient stale data utilization
- Enables accurate off-policy metrics monitoring

**Bypass mode** (sets $\pi_{\text{old}} = \pi_{\text{rollout}}$):
- Uses $\pi_{\text{rollout}}$ as both behavior policy and proximal policy (mathematically correct)
- Computational efficiency: Skips separate `old_log_prob` computation
- Does not achieve batch size invariance (proximal policy depends on data collection)

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
  - Introduced decoupled PPO: separating proximal policy (for controlling policy update size) from behavior policy (for off-policy correction) to achieve batch size invariance
- **Liu, J., Li, Y., et al. (2025).** "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
  - Blog post: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda
