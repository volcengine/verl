#!/usr/bin/env bash
# Example: Basic PPO training with Rollout Correction
# This demonstrates the standard setup for correcting distribution mismatch

set -xeuo pipefail

# ==============================================================================
# Rollout Correction Configuration
# ==============================================================================

# Importance Sampling (IS) weights configuration
rollout_is="token"                        # "token", "sequence", or null to disable
rollout_is_threshold=2.0                  # Upper threshold for IS weights

# Rejection Sampling (RS) configuration
rollout_rs="null"                         # "token", "sequence", "geometric", or null to disable
rollout_rs_threshold="null"               # RS upper threshold
rollout_rs_threshold_lower="null"         # RS lower threshold

# Veto mechanism (optional, independent of IS/RS)
rollout_token_veto_threshold="null"       # Per-token veto threshold (null to disable)

# ==============================================================================
# Model and Data Configuration
# ==============================================================================

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
TRAIN_FILE=${TRAIN_FILE:-"data/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/test.parquet"}

max_prompt_length=512
max_response_length=1024

# ==============================================================================
# Training Configuration
# ==============================================================================

train_batch_size=128
ppo_mini_batch_size=32
ppo_epochs=1
learning_rate=5e-7

# ==============================================================================
# Algorithm Configuration
# ==============================================================================

adv_estimator=gae
gamma=1.0
lam=0.95

# ==============================================================================
# Launch Training
# ==============================================================================

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.gamma=${gamma} \
    algorithm.lam=${lam} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name=vllm \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="rollout_corr_example" \
    trainer.experiment_name="basic_token_truncate" \
    trainer.total_epochs=10

echo "Training completed!"
echo ""
echo "Rollout Correction Configuration:"
echo "  - IS weights: ${rollout_is}"
echo "  - IS threshold: ${rollout_is_threshold}"
echo "  - RS mode: ${rollout_rs}"
echo "  - Veto threshold: ${rollout_token_veto_threshold}"
echo ""
echo "Monitor these key metrics in wandb:"
echo "  - rollout_corr/rollout_is_mean (should be ~1.0)"
echo "  - rollout_corr/rollout_is_eff_sample_size (should be >0.5)"
echo "  - rollout_corr/rollout_is_veto_fraction (should be <0.1)"
