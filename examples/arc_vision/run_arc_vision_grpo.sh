#!/bin/bash
# Arc Vision RL Training Script using GRPO
# This script properly configures VERL with all necessary overrides

set -ex

# Data paths
TRAIN_DATA=${TRAIN_DATA:-"/root/data/arc_vision/screenspot/train.parquet"}
VAL_DATA=${VAL_DATA:-"/root/data/arc_vision/screenspot/validation.parquet"}

# Model paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/arc_vision"}

# Base directory
BASE_DIR=${BASE_DIR:-"/root/verl"}

# Tool config path (absolute path for container)
TOOL_CONFIG_PATH="examples/arc_vision/config/tool_config/arc_vision_tools.yaml"

# Custom reward function path (absolute path for container)
REWARD_FUNCTION_PATH="examples/arc_vision/arc_vision_custom_reward.py"

# Configuration path
CONFIG_PATH="$BASE_DIR/examples/arc_vision/config"

# Number of GPUs
N_GPUS=${N_GPUS:-2}

# Launch VERL PPO training with GRPO algorithm  
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='arc_vision_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.image_key=images \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.reward_fn_key=ground_truth \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.02 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.strategy=fsdp \
    critic.optim.lr=0.0 \
    critic.model.path="$MODEL_PATH" \
    \
    reward_model.enable=false \
    reward_model.reward_manager=naive \
    \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=arc_vision_compute_reward \
    \
    trainer.total_epochs=5 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=arc_vision_rl \
    trainer.experiment_name=qwen2.5_vl_3b_screenspot_grpo \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    $@