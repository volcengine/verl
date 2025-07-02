#!/bin/bash
# Simplified Arc Vision RL Training Script
# This version shows the essential parameters clearly

set -ex

# Essential paths (adjust these for your setup)
TRAIN_DATA="/root/data/arc_vision/screenspot/train.parquet"
VAL_DATA="/root/data/arc_vision/screenspot/validation.parquet"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="outputs/arc_vision"

# Run VERL training with GRPO
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.image_key=images \
    data.reward_fn_key=ground_truth \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="examples/arc_vision/config/tool_config/arc_vision_tools.yaml" \
    \
    reward_model.enable=false \
    \
    custom_reward_function.path="examples/arc_vision/arc_vision_custom_reward.py" \
    custom_reward_function.name=arc_vision_compute_score_fn \
    \
    trainer.total_epochs=5 \
    trainer.project_name=arc_vision_rl \
    trainer.experiment_name=qwen2.5_vl_3b_screenspot_grpo \
    trainer.n_gpus_per_node=2 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    $@