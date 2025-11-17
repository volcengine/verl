# Mathematical Formulations of Rollout Correction Methods in `verl`

**Author:** [Yingru Li](https://richardli.xyz)
**Last updated:** 2025-11-04

---

> **📖 Documentation Structure**
> - **This document** - Mathematical theory: formulations, derivations, and algorithmic foundations
> - **[Rollout Correction Usage Guide](rollout_corr.md)** - Practical implementation: configurations, presets, troubleshooting
>
> Start here for theory and design rationale, refer to the usage guide for implementation.

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
3. [Algorithmic Components and Combinations](#3-algorithmic-components-and-combinations)
4. [Off-Policy Diagnostic Metrics](#4-off-policy-diagnostic-metrics)
5. [Summary and Decision Guide](#5-summary-and-decision-guide)
6. [Implementation References](#6-implementation-references)

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

**Implementation in verl:** The `pg_is` method implements off-policy REINFORCE with truncated importance sampling.

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
- $w_t = \frac{\pi_{\text{prox}}(a_t|s_t)}{\mu(a_t|s_t)}$: Importance sampling weight (corrects for behavior policy $\mu$). Here $\pi_{\text{prox}}$ is frozen during training, so $w_t$ is constant (no stopgrad operator needed).
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

## 3. Algorithmic Components and Combinations

The rollout correction framework in `verl` is built from **orthogonal components** that can be combined flexibly:

1. **Operating Mode**: How $\pi_{\text{old}}$ is computed (Decoupled vs Bypass)
2. **Loss Function**: PPO (with clipping) vs Pure IS (policy gradient only)
3. **IS/RS Aggregation Level**: Token, Sequence, or Geometric
4. **Safety Mechanisms**: Veto for catastrophic outliers

This section explains each component and their valid combinations.

### 3.1 Operating Modes: Decoupled vs Bypass

The operating mode determines how the proximal policy $\pi_{\text{old}}$ is computed.

#### 3.1.1 Decoupled Mode (Three Policies)

**Configuration:** `bypass_mode = false`

**Policy setup:**
- $\pi_{\text{rollout}}$: Behavior policy (data collection)
- $\pi_{\text{old}}$: Proximal policy (computed via `actor.compute_log_prob()` at start of training epoch)
- $\pi_{\theta}$: Current policy (being updated)

**IS ratio:** $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (corrects Drift 1: rollout→old)

**PPO ratio:** $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ (corrects Drift 2: old→current)

**Properties:**
- ✅ Achieves batch size invariance
- ✅ Separately corrects two distribution drifts
- ✅ Efficient stale data utilization
- ❌ Extra forward pass needed (`actor.compute_log_prob()`)

#### 3.1.2 Bypass Mode (Two Policies)

**Configuration:** `bypass_mode = true`

**Policy setup:**
- $\pi_{\text{rollout}}$: Behavior policy (data collection)
- $\pi_{\text{old}} = \pi_{\text{rollout}}$: Proximal policy equals behavior policy
- $\pi_{\theta}$: Current policy (being updated)

**Ratios:**
- **With PPO loss** (`use_policy_gradient = false`): No separate IS computation; PPO ratio $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ clips against rollout policy
- **With policy gradient loss** (`use_policy_gradient = true`): IS ratio $\rho_t = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ computed on-the-fly in loss function

**Properties:**
- ✅ Skips `actor.compute_log_prob()` call (faster)
- ✅ Handles off-policy correction via IS/RS (when using policy gradient with IS/RS)
- ✅ Uses two policies instead of three (π_rollout = π_old)
- ⚠️ Does not separate proximal policy from behavior policy (unlike decoupled mode)

---

### 3.2 Loss Functions: PPO vs Policy Gradient

#### 3.2.1 PPO Loss (with Clipping)

**Configuration:** `use_policy_gradient = false`

**Loss function:**

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $w_t$: IS weight (depends on aggregation level, see Section 3.3). In decoupled mode, $w_t = \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}$ where $\pi_{\text{old}}$ is frozen, so $w_t$ is constant (no stopgrad needed). In bypass mode with PPO loss, no separate IS weights are typically computed.
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: PPO ratio
- $\epsilon$: Clip range (typically 0.2)

**Properties:**
- Trust region control via clipping
- Limits policy update magnitude
- Standard in RL training

#### 3.2.2 Policy Gradient Loss (with IS/RS Correction)

**Configuration:** `use_policy_gradient = true` (requires `bypass_mode = true`)

**Loss function** (example with sequence-level IS):

$$
L_{\text{PG}}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ \text{stopgrad}(w_{\text{seq}}(\theta)) \cdot \sum_{t \in T} \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

where:
- $w_{\text{seq}}(\theta)$: Sample weight (IS or RS, see §3.3-3.4 for details)
- For IS: $w_{\text{seq}}(\theta) = \min\left( \prod_{t \in T} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, C_{\text{IS}} \right)$
- For RS: $w_{\text{seq}}(\theta) \in \{0, 1\}$ (binary rejection mask)
- **stopgrad operator**: The weight $w_{\text{seq}}(\theta)$ is computed using $\pi_\theta$ but treated as a **constant coefficient** when computing $\nabla_\theta L$. This is essential for importance sampling correctness (see theoretical justification below).

**Effective gradient:**

$$
\nabla_\theta L_{\text{PG}} = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ \text{stopgrad}(w_{\text{seq}}(\theta)) \cdot \sum_{t \in T} \nabla_\theta \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

**Theoretical Justification for stopgrad:**

The stopgrad operator is **mathematically required** by importance sampling theory, not an implementation detail. Here's why:

**The fundamental principle**: Importance sampling is a technique to **change the measure** (reweight samples from one distribution to estimate expectations under another), not to optimize the reweighting function itself.

**Formal derivation**:

1. **Original objective**: We want to optimize $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t A_t]$.

2. **Off-policy setting**: We only have samples from $\pi_{\text{rollout}}$, so we use importance sampling:
   $$
   J(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{rollout}}} \left[ \underbrace{\frac{P_{\pi_\theta}(\tau)}{P_{\pi_{\text{rollout}}}(\tau)}}_{w(\tau;\theta)} \sum_t A_t \right]
   $$

3. **Computing the policy gradient**: The correct gradient uses the **policy gradient theorem BEFORE importance sampling**:
   $$
   \begin{aligned}
   \nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t A_t\right] \\
   &= \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \quad \text{(policy gradient theorem)} \\
   &= \mathbb{E}_{\tau \sim \pi_{\text{rollout}}} \left[ w(\tau;\theta) \sum_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \quad \text{(change of measure)}
   \end{aligned}
   $$

   In the final line, $w(\tau;\theta)$ appears as a **multiplicative coefficient** from the change of measure, not as something we differentiate.

4. **What goes wrong without stopgrad**: If we naively compute $\nabla_\theta \left[w(\theta) \log \pi_\theta \right]$ in the loss, we get:
   $$
   \nabla_\theta \left[w(\theta) \log \pi_\theta \right] = \underbrace{\log \pi_\theta \cdot \nabla_\theta w(\theta)}_{\text{WRONG: bias term}} + \underbrace{w(\theta) \cdot \nabla_\theta \log \pi_\theta}_{\text{CORRECT: IS-weighted gradient}}
   $$

   The first term $\log \pi_\theta \cdot \nabla_\theta w(\theta)$ is an artifact of the computational trick (using loss times log-prob), not part of the true policy gradient. It biases the gradient estimator and optimizes a different objective than $J(\theta)$.

5. **Implementation requirement**: In PyTorch, to compute only the second term, we must use:
   ```python
   loss = -advantages * log_prob * rollout_is_weights.detach()  # stopgrad on weights
   ```
   Without `.detach()`, autograd computes both terms, giving an incorrect gradient.

**Intuition**: The IS weight $w(\theta)$ tells us "how much to trust this sample" for estimating the gradient under $\pi_\theta$. We update $\theta$ to maximize the reweighted objective, but we don't update $\theta$ to maximize the weight itself—that would be circular reasoning (optimizing the correction factor instead of the actual objective).

**Properties:**
- **Algorithm**: Off-policy REINFORCE + IS/RS correction
- **No PPO clipping**: Pure policy gradient
- **Always uses bypass mode**: Direct $\pi_\theta$ to $\pi_{\text{rollout}}$ comparison
- **Fast**: Single forward pass

**Implementation:** `compute_policy_loss_with_rollout_correction()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L1537-L1681)

---

### 3.3 IS/RS Aggregation Levels

The aggregation level determines how per-token probability ratios are combined into IS weights and/or rejection masks. This choice is **orthogonal to the operating mode** - you can use any aggregation level in either decoupled or bypass mode.

#### 3.3.1 Token-Level Aggregation

**IS weights:** $w_t = \min(\rho_t, C_{\text{IS}})$ where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (decoupled) or $\rho_t = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (bypass/pure IS)

**Configuration:**
```python
rollout_is = "token"  # IS weights
rollout_rs = "token"  # Optional: rejection sampling
```

**Properties:**
- Independent truncation per token
- Lower variance than sequence-level (product of ratios bounded individually)
- Typical threshold: 1.5 - 5.0
- Optional batch normalization (§3.6): Normalizes over all token weights to ensure $\mathbb{E}[\tilde{w}_t] = 1$ (reduces variance)

**Loss function (REINFORCE + Token IS):**

$$
L_{\text{REINFORCE+TIS}}(\theta) = -\mathbb{E}_t \left[ \text{stopgrad}(w_t) \cdot \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $w_t = \min(\rho_t, C_{\text{IS}})$ are the truncated token-level IS weights. The stopgrad operator ensures that when computing $\nabla_\theta L$, the weights are treated as constants (see §3.2.2 for theoretical justification). This formulation can also be combined with PPO clipping by replacing the REINFORCE gradient with the clipped surrogate objective.

**Implementation:**
- IS weights: `compute_rollout_correction_weights()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L325-L402)
- Loss: `compute_policy_loss()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L812-L884)

#### 3.3.2 Sequence-Level Aggregation

**IS weights:** $w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, C_{\text{IS}} \right) = \min\left( \exp\left(\sum_{t \in T} \log \rho_t\right), C_{\text{IS}} \right)$ (broadcast to all tokens)

**Configuration:**
```python
rollout_is = "sequence"  # IS weights
rollout_rs = "sequence"  # Optional: rejection sampling
```

**Properties:**
- Multiplicative aggregation across sequence
- More sensitive to outliers than token-level
- Typical threshold: 2.0 - 10.0
- Optional batch normalization (§3.6): Normalizes over sequence means (one weight per sequence)

**Loss function (REINFORCE + Sequence IS):**

$$
L_{\text{REINFORCE+SeqIS}}(\theta) = -\mathbb{E}_t \left[ \text{stopgrad}(w_{\text{seq}}) \cdot \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $w_{\text{seq}}$ is broadcast to all tokens in the sequence. The stopgrad operator ensures correct IS gradient computation (see §3.2.2). This formulation can also be combined with PPO clipping.

#### 3.3.3 Geometric Aggregation

**IS weights (for rejection only):** $\rho_{\text{geo}} = \exp\left( \frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{1/|T|}$ (broadcast to all tokens)

**Configuration:**
```python
rollout_is = null  # No IS weights, pure rejection
rollout_rs = "geometric"  # Rejection sampling only
```

**Properties:**
- Geometric mean of per-token ratios
- More sensitive than arithmetic product (sequence-level)
- Typical threshold: 1.0001 - 1.001 (tighter than sequence/token level)
- **Used for rejection sampling only, not IS weighting**

**Why tight thresholds?**
For 100 tokens with $\rho_t = 1.01$ each:
- Arithmetic product: $\prod_{t=1}^{100} \rho_t = 1.01^{100} \approx 2.7$
- Geometric mean: $(1.01)^{1} = 1.01$

A threshold of 1.001 means rejecting sequences with average per-token deviation > 0.1%.

**Loss function (REINFORCE + Geometric RS):**

$$
L_{\text{GeoRS}}(\theta) = -\mathbb{E}_{(s,a) \mid \text{seq} \in \mathcal{A}_{\text{geo}}} \left[ \sum_{t \in T} \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $\mathcal{A}_{\text{geo}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \rho_{\text{geo}} \leq C_{\text{RS-upper}} \}$ is the acceptance set (rejection mask). No IS weights are used, so no stopgrad needed. This formulation can also be combined with PPO clipping.

---

### 3.4 Rejection Sampling (RS)

Rejection sampling can be added to **any combination** of operating mode and aggregation level. It modifies the `response_mask` to exclude outlier tokens/sequences.

**Configuration:**
```python
rollout_rs = "token"  # or "sequence" or "geometric"
rollout_rs_threshold = 2.0  # Upper threshold
rollout_rs_threshold_lower = 0.5  # Lower threshold (auto-reciprocal if null)
```

**Acceptance set:**
- **Token-level**: $\mathcal{A}_{\text{token}} = \{ t : C_{\text{RS-lower}} \leq \rho_t \leq C_{\text{RS-upper}} \}$
- **Sequence-level**: $\mathcal{A}_{\text{seq}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \prod_{t \in T} \rho_t \leq C_{\text{RS-upper}} \}$
- **Geometric**: $\mathcal{A}_{\text{geo}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \rho_{\text{geo}} \leq C_{\text{RS-upper}} \}$

**Properties:**
- Separate from IS weighting (can use RS without IS)
- Reduces effective sample size
- Filters extreme outliers

**Implementation:** `compute_rollout_rejection_mask()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L80-L188)

---

### 3.5 Veto Mechanism

An **independent** safety layer that rejects sequences with catastrophically low token probabilities.

**Configuration:**
```python
rollout_token_veto_threshold = 1e-4  # null = disabled
```

**Veto condition:**

$$
\text{Reject entire sequence if } \exists t \in T \text{ such that } \rho_t < C_{\text{veto}}
$$

**Properties:**
- Prevents catastrophic updates from tokens with near-zero probability
- **Independent** of IS/RS settings (always applied if enabled)
- Checks **unclamped per-token ratios** before safety bounds
- Typical values: $10^{-4}$ to $10^{-6}$

**Implementation:** [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L620-L640)

---

### 3.6 Batch Normalization

An optional variance reduction technique that normalizes IS weights to have mean 1.0 within each batch.

**Configuration:**
```python
rollout_is_batch_normalize = True  # Default: False
```

**Normalization formula (aggregation-aware):**

For **token-level IS** (§3.3.1):

$$
\tilde{w}_t = \frac{w_t}{\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i,t} w_{i,t} \cdot m_{i,t}}
$$

where $w_{i,t}$ are truncated token IS weights, $m_{i,t}$ is the response mask, and normalization is over **all tokens**.

For **sequence-level IS** (§3.3.2):

$$
\tilde{w}_i = \frac{w_i}{\frac{1}{B}\sum_{j=1}^B \bar{w}_j}
$$

where $\bar{w}_j = \frac{1}{T_j}\sum_{t=1}^{T_j} w_{j,t} \cdot m_{j,t}$ is the per-sequence mean (all tokens in a sequence have the same weight), and normalization is over **sequences**.

**Properties:**
- Applied **after** truncation to preserve truncation semantics
- Ensures $\mathbb{E}[\tilde{w}] = 1$ within each batch
- **Aggregation-aware**: Token-level normalizes over tokens; sequence-level normalizes over sequences
- Uses `masked_mean` to respect padding tokens
- Reduces gradient magnitude variance by removing random batch-level scale fluctuations

**Metrics:**
- `rollout_is_batch_norm_factor`: The normalization factor applied (batch mean before normalization)

**Implementation:** [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L401-L421)

---

### 3.7 Combination Matrix

#### Available Preset Methods

| Preset Method | Mode | IS Level | RS Level | Properties |
|---------------|------|----------|----------|------------|
| `decoupled_token_is()` | Decoupled | token | - | Per-token IS weights |
| `decoupled_seq_is()` | Decoupled | sequence | - | Sequence-level IS weights |
| `decoupled_seq_is_rs()` | Decoupled | sequence | sequence | Sequence IS + sequence RS |
| `decoupled_geo_rs()` | Decoupled | - | geometric + veto | Geometric RS + veto, no IS weights |
| `ppo_is_bypass()` | Bypass | - | - | Bypass mode, skips old_log_prob |
| `pg_rs()` | Bypass | - | geometric + veto | Policy gradient with RS (no IS weights) |
| `pg_is()` | Bypass | sequence | - | Policy gradient with IS |
| `disabled()` | - | - | - | Metrics only, no correction |

**Note:** All presets use PPO loss except `pg_is()` and `pg_rs()` which use policy gradient (both require `use_policy_gradient=True`).

#### Additional Supported Combinations (Manual Configuration)

These combinations are **fully supported** but require manual configuration:

**1. Token IS + Token RS**
```python
config = RolloutCorrectionConfig(
    rollout_is="token",
    rollout_is_threshold=2.0,
    rollout_rs="token",
    rollout_rs_threshold=2.0,
)
```
**Properties:** Token-level IS weights + token-level RS mask.

**2. Pure Token RS**
```python
config = RolloutCorrectionConfig(
    rollout_is=None,
    rollout_rs="token",
    rollout_rs_threshold=2.0,
)
```
**Properties:** Token-level RS mask only, no IS weights.

**3. Pure Sequence RS**
```python
config = RolloutCorrectionConfig(
    rollout_is=None,
    rollout_rs="sequence",
    rollout_rs_threshold=2.0,
)
```
**Properties:** Sequence-level RS mask only, no IS weights.

**Key properties:**
- Any IS aggregation level (token/sequence) can be used in either decoupled or bypass mode
- Rejection sampling can be added to any combination
- Veto is independent and can be added to any combination
- Geometric aggregation is typically used for RS only (not IS weighting)
- Pure RS (`pg_rs`) uses bypass + geometric RS with `use_policy_gradient=True` for pure policy gradient (no IS weights)
- All combinations in the table above are valid and supported by the implementation

---

### 3.8 Common Implementation Mistake

#### Incorrect LLM-RL Implementation (PPO Without Rollout Correction)

**Theory:** Naive LLM-RL implementation that incorrectly applies PPO by **ignoring the actual rollout policy** and assuming $\pi_{\text{old}} = \pi_{\text{rollout}}$.

**Note:** This incorrect implementation pattern was identified in [Liu, Li, et al. (2025)](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) as a key cause of training instability in LLM-RL systems, motivating the development of this rollout correction framework.

**Loss Function:**

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ (ignores $\pi_{\text{rollout}}$).

**Why it's wrong:**
- **Ignores $\pi_{\text{rollout}}$**: Uses $\pi_{\text{old}}$ as behavior policy instead of actual $\pi_{\text{rollout}}$
- **Policy mismatch**: In LLM-RL, rollout typically uses different precision/backend/checkpoint than training, causing $\pi_{\text{rollout}} \neq \pi_{\text{old}}$ even with same model weights
- **Not PPO's fault**: PPO itself is correct; the issue is the incorrect assumption

**Correct alternatives:**
1. **Decoupled mode**: Three policies with IS correction from $\pi_{\text{rollout}}$ to $\pi_{\text{old}}$
2. **Bypass mode**: Two policies using $\pi_{\text{rollout}}$ as both behavior policy and proximal policy
3. **Bypass + Policy Gradient mode**: Two policies with IS/RS correction and no PPO clipping

**Implementation:** `compute_policy_loss()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L812-L884)

---

## 4. Off-Policy Diagnostic Metrics

These metrics quantify the severity of off-policy drift.

**Note on notation:** Metrics use $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$. In bypass mode, $\pi_{\text{old}} = \pi_{\text{rollout}}$, so metrics measure rollout→current drift using $\rho_t = \frac{\pi_{\theta}}{\pi_{\text{rollout}}}$ instead.

### 4.1 KL Divergence

**Direct KL estimator:**

$$
\text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}}) = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \log \pi_{\text{rollout}}(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t) \right]
$$

**K3 KL estimator** (alternative formulation):

$$
\text{KL}_{\text{K3}} = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t - \log \rho_t - 1 \right]
$$

where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$.

### 4.2 Perplexity

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

### 4.3 Chi-squared Divergence

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

## 5. Summary and Decision Guide

### 5.1 Method Summary Table

| Method | Theory | Policies | PPO Clip | IS Correction | Correctness | Speed |
|--------|--------|----------|----------|---------------|-------------|-------|
| `pg_is` | Off-policy REINFORCE | 2 (rollout, θ) | ❌ | ✅ Seq-level | ✅ Correct | **Fast** |
| `pg_rs` | Pure PG + Geo RS | 2 (rollout, θ) | ❌ | Rejection only | ✅ Correct | **Fast** |
| Naive LLM-RL | Incorrect PPO usage | 2 (old, θ) | ✅ | ❌ | ⚠️ Incorrect | Standard |
| `ppo_is_bypass` | PPO (rollout as prox) | 2 (rollout, θ) | ✅ | ❌ | ✅ Correct | **Fast** |
| `decoupled_token_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Token-level | ✅ Correct | Standard |
| `decoupled_seq_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Seq-level | ✅ Correct | Standard |
| `decoupled_seq_is_rs` | Decoupled PPO + RS | 3 (rollout, old, θ) | ✅ | ✅ + Rejection | ✅ Correct | Standard |
| `decoupled_geo_rs` | Decoupled PPO + Geo RS | 3 (rollout, old, θ) | ✅ | Rejection only | ✅ Correct | Standard |

### 5.2 Method Characteristics by Scenario

**Off-policy severity:**
- **Negligible** (same checkpoint, minor differences): `ppo_is_bypass` uses $\pi_{\text{rollout}}$ as proximal policy (mathematically correct); naive LLM-RL implementations use $\pi_{\text{old}}$ instead of $\pi_{\text{rollout}}$ (mathematically incorrect when $\pi_{\text{rollout}} \neq \pi_{\text{old}}$)
- **Moderate** (async workers, slight staleness): `decoupled_token_is` provides per-token IS correction with separate proximal policy
- **Severe** (replay buffers, old data): `decoupled_seq_is` and `decoupled_seq_is_rs` provide sequence-level IS correction with optional rejection sampling

**Algorithm properties:**
- **Batch size invariance**: Decoupled mode with three policies (`decoupled_token_is`, `decoupled_seq_is`) achieves batch size invariance
- **Computational efficiency**: Bypass mode (`ppo_is_bypass`) skips `old_log_prob` computation
- **Pure policy gradient**: `pg_is` implements off-policy REINFORCE without PPO clipping

### 5.3 Decoupled Mode vs Bypass Mode

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

## 6. Implementation References

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
