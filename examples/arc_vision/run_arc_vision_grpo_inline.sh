#!/bin/bash
# Arc Vision RL training with inline configuration (no config file needed)
# This approach uses all parameter overrides directly

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

# Launch from VERL root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.."

# Run with all parameters inline (no custom config file needed)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.image_key=images \
    data.reward_fn_key=ground_truth \
    data.data_source=arc_vision \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.gradient_accumulation_steps=32 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_loss_coef=0.02 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=2 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/arc_vision/config/tool_config/arc_vision_tools.yaml \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.adv_norm=true \
    algorithm.use_kl_in_reward=False \
    critic.strategy=fsdp \
    critic.optim.lr=0.0 \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    reward_model.enable=false \
    reward_model.reward_manager=naive \
    custom_reward_function.path=examples/arc_vision/arc_vision_custom_reward.py \
    custom_reward_function.name=arc_vision_compute_score_fn \
    custom_reward_function.reward_kwargs.confidence_threshold=0.7 \
    custom_reward_function.reward_kwargs.reward_weights.task=0.6 \
    custom_reward_function.reward_kwargs.reward_weights.tool=0.3 \
    custom_reward_function.reward_kwargs.reward_weights.gate=0.1 \
    custom_reward_function.reward_kwargs.tool_penalties.unnecessary_tool=-0.5 \
    custom_reward_function.reward_kwargs.tool_penalties.missed_opportunity=-0.3 \
    custom_reward_function.reward_kwargs.tool_penalties.ineffective_tool=-0.2 \
    custom_reward_function.reward_kwargs.tool_penalties.excessive_tools=-0.4 \
    trainer.total_epochs=5 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='arc_vision_rl' \
    trainer.experiment_name='qwen2.5_vl_3b_screenspot_grpo' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=outputs/arc_vision $@