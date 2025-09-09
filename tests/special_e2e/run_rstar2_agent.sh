#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/rstar2_agent/src/config"
NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

train_traj_micro_bsz_per_gpu=2 # b
n_resp_per_prompt=8 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

exp_name="$(basename "${MODEL_ID,,}")-rstar2-agent-minimal"

python3 -m recipe.rstar2_agent.src.main_rstar2_agent \
    --config-path="$CONFIG_PATH" \
    --config-name='rstar2_agent_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    algorithm.use_kl_in_reward=False \
    augmentation.do_down_sampling=True \
    augmentation.down_sampling_config.reject_equal_reward=True \
    augmentation.down_sampling_config.roc_error_ratio=True \
    augmentation.down_sampling_config.roc_answer_format=True \
    augmentation.down_sampling_config.min_zero_reward_trace_num=1 \
    augmentation.down_sampling_config.min_non_zero_reward_trace_num=1 \
    augmentation.down_sampling_config.down_sample_to_n=4 \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl-test' \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    data.train_files="${HOME}/data/gsm8k/train.parquet" \
    data.val_files="${HOME}/data/gsm8k/test.parquet" \
    trainer.total_epochs=2 $@
