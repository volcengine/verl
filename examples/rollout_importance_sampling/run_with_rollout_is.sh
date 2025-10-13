#!/usr/bin/env bash
# Example: Basic PPO training with Rollout Importance Sampling
# This demonstrates the standard setup for correcting distribution mismatch

set -xeuo pipefail

# ==============================================================================
# Rollout Importance Sampling Configuration
# ==============================================================================

# Main control: Upper threshold for IS weights (null = disabled, float = enabled)
rollout_is_threshold=2.0

# Whether to apply IS weights to policy loss
# true = apply weights to loss, false = compute metrics only
rollout_is=true

# Lower threshold (null = auto-reciprocal, i.e., 1/upper = 0.5)
rollout_is_threshold_lower=null

# Aggregation level: token | sequence | geometric (experimental)
rollout_is_level=token

# Bounding mode: truncate (cap upper) | mask (zero outside bounds)
rollout_is_mode=truncate

# Catastrophic outlier veto threshold
rollout_is_veto_threshold=1e-4

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
    algorithm.rollout_is=${rollout_is} \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    algorithm.rollout_is_level=${rollout_is_level} \
    algorithm.rollout_is_mode=${rollout_is_mode} \
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name=vllm \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="rollout_is_example" \
    trainer.experiment_name="basic_token_truncate" \
    trainer.total_epochs=10

echo "Training completed!"
echo ""
echo "Rollout IS Configuration:"
echo "  - Threshold: ${rollout_is_threshold}"
echo "  - Apply to loss: ${rollout_is}"
echo "  - Level: ${rollout_is_level}"
echo "  - Mode: ${rollout_is_mode}"
echo ""
echo "Monitor these key metrics in wandb:"
echo "  - mismatch/rollout_is_mean (should be ~1.0)"
echo "  - mismatch/rollout_is_eff_sample_size (should be >0.5)"
echo "  - mismatch/rollout_is_veto_fraction (should be <0.1)"
