# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

set -x

ulimit -n 65535

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd -P)
PROJECT_DIR=$SCRIPT_DIR/..
CONFIG_PATH="$PROJECT_DIR/src/config"
PROJECT_NAME="rstar2-agent"
EXPERIMENT_NAME="eval-rstar2-agent-math500"

python3 -m recipe.rstar2_agent.src.main_rstar2_agent \
    --config-path="$CONFIG_PATH" \
    --config-name='rstar2_agent_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=30720 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    algorithm.use_kl_in_reward=False \
    augmentation.do_down_sampling=True \
    augmentation.down_sampling_config.reject_equal_reward=True \
    augmentation.down_sampling_config.roc_error_ratio=True \
    augmentation.down_sampling_config.roc_answer_format=True \
    augmentation.down_sampling_config.min_zero_reward_trace_num=2 \
    augmentation.down_sampling_config.min_non_zero_reward_trace_num=2 \
    augmentation.down_sampling_config.down_sample_to_n=16 \
    reward_model.reward_manager=code_judge \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_training_steps=200 \
    trainer.val_only=True \
    data.train_files="['$HOME/data/rstar2-agent/dapo-math-17k-en/train.parquet']" \
    data.val_files="['$HOME/data/rstar2-agent/math500/test.parquet']" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/src/config/tool_config/python_tool_config.yaml" \
    trainer.total_epochs=15 $@ 2>&1 | tee $PROJECT_NAME-$EXPERIMENT_NAME.log
