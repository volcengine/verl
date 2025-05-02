# SPIN (Self-Play Fine-Tuning) Inspired Recipe for Verl

## Overview

This recipe implements an **Online DPO (Direct Preference Optimization)** algorithm, drawing inspiration from concepts seen in **Self-Play Fine-Tuning (SPIN)**, adapted to run within the `verl` Reinforcement Learning framework.

It provides an alternative to Proximal Policy Optimization (PPO) for fine-tuning language models. Instead of maximizing a scalar reward signal, this approach directly optimizes the policy to align with *preference data*. This preference data indicates which of two responses to a given prompt is considered "better".

In this specific implementation, the preference data is generated *online*:
1. The current policy model generates two (or more) responses for each prompt.
2. A reward model or reward function evaluates these responses to determine which one is preferred (chosen) and which is dispreferred (rejected).
3. This preference pair (`prompt`, `chosen_response`, `rejected_response`) is used to update the policy model using the DPO loss function.

## Background & Papers

The core concepts underpinning this recipe are:

1.  **Direct Preference Optimization (DPO):**
    * DPO provides a mechanism to directly optimize a language model on preference data without needing to first train a separate reward model (though one can be used for *labeling* preferences). It leverages a theoretical connection between reward functions and optimal policies.
    * The loss function contrasts the likelihood of the chosen vs. rejected response under the policy being trained and a reference policy.
    * **Paper:** [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)

2.  **Self-Play Fine-Tuning (SPIN):**
    * SPIN proposes an iterative method where a model generates its *own* training data. In one variant, the model generates a response, and then a previous iteration of the model (acting as the "reference" or "teacher") generates another response. The original response is treated as "good" (akin to chosen) and the teacher's response as "bad" (akin to rejected), forming data for an SFT-like loss.
    * While this recipe primarily uses the DPO loss, the online generation loop where the current model generates data used for its own update shares conceptual similarities with the self-play idea in SPIN. The periodic update of the reference model further aligns with iterative self-improvement concepts.
    * **Paper:** [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) (Chen et al., 2024)

**This Recipe's Approach:** Combines the DPO loss formulation with an online data generation loop. Preferences between generated pairs are determined using a provided reward source, and an explicit reference model (potentially a periodically updated copy of the actor) is used within the DPO loss calculation.

## Implementation in Verl

This implementation adapts the existing PPO infrastructure provided by `verl`:

* **No Critic:** The value function critic model used in PPO is not required and not used.
* **Reference Model:** An explicit reference policy model (`ref_policy_wg`) is used. This is essential for the DPO loss calculation. This implementation includes logic to potentially update this reference model periodically from the main actor model's weights (controlled by `ref_update_freq` in the config).
* **Preference Calculation:** Logic (`compute_onlineDPO_pref` in `core_algos.py`) is added to determine chosen/rejected pairs based on scores from a reward source.
* **DPO Loss:** The PPO policy loss and advantage calculations are replaced with the DPO loss computation (`compute_online_dpo_loss` in `core_algos.py`) within the actor update step (`dp_actor.py`).
* **Training Loop:** The `SpinTrainer` (in `spin_trainer.py`, analogous to `RayPPOTrainer`) orchestrates the cycle of generation, preference labeling, reference model updates (if configured), and policy updates using the DPO loss.

## Configuration

* Primary configuration is handled in `config/spin_trainer.yaml`.
* Key settings include:
    * `data`: Paths to training/validation prompt files, batch sizes, sequence lengths.
    * `actor_rollout_ref`: Paths to the base model (used for actor and reference), FSDP settings, optimization parameters (learning rate).
    * `reward_model`: Configuration for the reward model used for preference labeling (if applicable).
    * `algorithm`: DPO-specific hyperparameters like `dpo_beta`, `dpo_loss_type`.
    * `trainer`: Distributed training settings (nodes, GPUs), logging, checkpointing frequency, and the `ref_update_freq` parameter (set > 0 to enable periodic reference model updates).

## Usage

1.  **Configure:** Modify `config/spin_trainer.yaml` with desired model paths, data paths, hyperparameters, and distributed settings.
2.  **Set Environment:** Ensure necessary environment variables (CUDA device visibility, paths, API keys like `WANDB_API_KEY` if using WandB) are set correctly. The `run_spin.sh` script provides examples.
3.  **Launch:** Execute the provided launch script:
    ```bash
    bash run_spin.sh
    ```
    This script handles setting `CUDA_VISIBLE_DEVICES` and runs the main training script (`main_spin.py`) with appropriate overrides.

## Key Files

* `main_spin.py`: Main entry point using Hydra to load config and launch the `SpinTrainer`.
* `spin_trainer.py`: Contains the `SpinTrainer` class (likely analogous to `RayPPOTrainer`) orchestrating the online DPO training loop.
* `fsdp_workers.py`: Implements the Ray workers (Actor, Reference) using FSDP.
* `dp_actor.py`: Contains the `DataParallelPPOActor` class, including the DPO-specific policy update method.
* `core_algos.py`: Includes helper functions for DPO loss and online preference calculation.
* `config/spin_trainer.yaml`: Main Hydra configuration file for this recipe.
* `run_spin.sh`: Example bash script for launching a training run.
* `README.md`: This file.