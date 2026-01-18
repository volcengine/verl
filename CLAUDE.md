# CLAUDE.md - RARO Implementation Guide (verl Framework)

This document provides the architectural blueprint, algorithmic logic, and implementation details for reproducing the paper **"Escaping the Verifier: Learning to Reason via Demonstrations" (RARO)** using the `verl` framework.

## 1. Paper Overview & Core Logic

**RARO (Relativistic Adversarial Reasoning Optimization)** is an Inverse Reinforcement Learning (IRL) approach. Unlike standard RLHF or RLVR (which use external verifiers), RARO trains a **Policy** to generate reasoning chains that can fool a **Relativistic Critic**, while simultaneously training the Critic to distinguish between Expert and Policy answers.

### 1.1 Key Architecture: Shared Weights
* **Single Model**: Use **one** LLM (initialized from an Instruct model, e.g., Qwen2.5-Instruct) to act as both Policy and Critic.
* **Role Switching**: Roles are switched dynamically via System Prompts.
* **Optimization**: GRPO (Group Relative Policy Optimization) updates both roles simultaneously.

### 1.2 The Adversarial Game
* **Policy Goal**: Generate $(z, a_{pol})$ that makes the Critic output `Policy` (deceived) or `Tie`.
* **Critic Goal**: Given a triplet $(q, a_{exp}, a_{pol})$, correctly identify the `Expert` answer or output `Tie` if indistinguishable.
* **Output Space**: The Critic generates a Chain-of-Thought followed by a label: $l \in \{\text{Expert}, \text{Policy}, \text{Tie}\}$.

### 1.3 Reward Matrix (Mutually Exclusive)
Rewards are assigned based on the Critic's predicted label $l$.

| Critic Prediction ($l$) | Reward for Critic ($R_{crit}$) | Reward for Policy ($R_{pol}$) |
| :--- | :--- | :--- |
| **Expert** (Correct) | **1.0** | **0.0** (Failed to deceive) |
| **Policy** (Deceived) | **0.0** | **1.0** (Success) |
| **Tie** (Ambiguous) | **$\tau_{crit}$** (e.g., 0.55) | **$\tau_{pol}$** (e.g., 0.6) |

*Note: The "Tie" option is crucial for training stability to prevent degeneracy.*

---

## 2. Implementation Strategy in `verl`

RARO requires modifying the standard PPO/GRPO loop to support **Dual-Pass Rollout** and **Replay Buffers**.

### 2.1 The Training Loop (Step-by-Step)

The `verl` rollout worker must perform the following sequence in a single iteration:

1.  **Policy Rollout (Generator Pass)**:
    * Input: Batch of questions $q$.
    * Prompt: Policy System Prompt.
    * Output: $a_{pol}$ (with reasoning trace).
    * *Action*: Store $(q, a_{exp}, a_{pol})$ into **Replay Buffer**.

2.  **Data Mixing**:
    * Create a Critic Batch $\mathcal{C}$:
        * 50% from the current step's fresh generations.
        * 50% sampled from the Replay Buffer (historical generations).
    * *Purpose*: Prevent catastrophic forgetting in the Critic.

3.  **Critic Rollout (Discriminator Pass)**:
    * Input: Mixed Batch $\mathcal{C}$ formatted as triplets.
    * Prompt: Critic System Prompt (Comparison task).
    * Output: Critic CoT + Label ($l$).

4.  **Reward Parsing & Assignment**:
    * Parse $l$ from Critic output.
    * Calculate $R_{pol}$ and $R_{crit}$ using the Reward Matrix.
    * *Constraint*: Filter out samples exceeding token limits (Over-length filtering).

5.  **Joint GRPO Update**:
    * Objective: $J(\theta) = \lambda_{pol} J_{pol}(\theta) + \lambda_{crit} J_{crit}(\theta) - \beta D_{KL}$
    * Note: Weights are typically skewed towards the Critic (e.g., 8:1).

### 2.2 Critical Components to Implement

#### A. Prompt Management
Define two distinct templates.
* **Policy**: Standard assistant prompt.
* **Critic**: "You are an expert critic... Compare Answer 1 and Answer 2... Output final judgment as [Answer 1], [Answer 2], or [Tie]."

#### B. Replay Buffer Class
* **Structure**: FIFO Queue or Reservoir Buffer.
* **Data**: Stores tuples of `(question_id, expert_answer, policy_answer)`.
* **Capacity**: Large enough to cover diverse history (e.g., 10k-50k samples).

#### C. Custom Reward Manager
Instead of a separate RM model, the Reward Manager in RARO is logic-based:
* It takes the **Critic's text output** as input.
* It applies Regex/Rule-based parsing to extract the label.
* It assigns scores based on the ground truth (which answer was actually expert vs policy).

---

## 3. Configuration & Hyperparameters

Use these reference values (based on DeepMath experiments in the paper) for `config.yaml`:

| Category | Parameter | Value | Notes |
| :--- | :--- | :--- | :--- |
| **Optimization** | `lr` | 1e-6 | Lower than standard SFT |
| | `kl_coeff` | 1e-3 | Keep policy close to Ref |
| **Rewards** | `tau_pol` (Policy Tie) | **0.6** | Encourages "good enough" answers |
| | `tau_crit` (Critic Tie) | **0.55** | Encourages admitting uncertainty |
| **Loss Weights** | `lambda_pol` | **1/9** | Policy learns slowly |
| | `lambda_crit` | **8/9** | Critic must be robust |
| **Rollout** | `n_rollouts` | 16 | For GRPO group estimation |
| **Training** | `epochs` | 1 | Online RL typically 1 epoch per batch |

---

## 4. Development Checklist

- [ ] **Data Loader**: Ensure dataset yields $(q, a_{expert})$.
- [ ] **Prompt Templates**: Implement Role-Playing prompts (Policy/Critic).
- [ ] **Replay Buffer**: Implement efficient buffer with sampling support.
- [ ] **Dual Rollout Logic**: Modify `verl` worker to support Generate -> Mix -> Judge flow.
- [ ] **Parser**: Implement robust parsing for Critic outputs (handle edge cases where Critic fails to follow format).
- [ ] **Loss Function**: Implement the weighted joint loss: $\lambda_{pol} L_{pol} + \lambda_{crit} L_{crit}$.
- [ ] **Sanity Check**:
    - Verify Critic accuracy starts near 50% (random) and improves.
    - Verify Policy reward increases over time.
    - Verify Tie rate stabilizes (approx 70% in paper).

## 5. Potential Pitfalls

* **Mode Collapse**: If the Critic always outputs "Tie", reduce `tau_crit`.
* **Format Errors**: If Critic outputs cannot be parsed, mask these samples (do not train on them).
* **OOM (Out of Memory)**: Shared weights mean context length doubles in the Critic phase (Question + Ans1 + Ans2 + Critic CoT). Adjust `batch_size` accordingly.