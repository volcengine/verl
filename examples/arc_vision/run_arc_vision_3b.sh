#!/bin/bash
# Arc Vision RL training script for Qwen2.5-VL-3B with confidence-gated tool learning

set -x

# Check if data exists
DATA_DIR="${HOME}/data/arc_vision/screenspot"
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: ScreenSpot data not found at ${DATA_DIR}"
    echo "Please run: cd examples/arc_vision && python prepare_screenspot_data.py"
    exit 1
fi

# Set engine (default to sglang for multi-turn support)
ENGINE=${1:-sglang}

# Launch Arc Vision RL training from VERL root directory
# This ensures Hydra can find both default configs and our custom config
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.."  # Go to VERL root

# Copy our config to the trainer config directory temporarily
cp "${SCRIPT_DIR}/config/arc_vision_grpo.yaml" verl/trainer/config/

python3 -m verl.trainer.main_ppo \
    --config-name arc_vision_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.02 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/arc_vision/config/tool_config/arc_vision_tools.yaml \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.lam=0.95 \
    reward_model.enable=true \
    reward_model.custom_reward_function.path=examples/arc_vision/arc_vision_custom_reward.py \
    reward_model.custom_reward_function.name=arc_vision_compute_reward \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='arc_vision_rl' \
    trainer.experiment_name='qwen2.5_vl_3b_screenspot_confidence_gated' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=outputs/arc_vision $@