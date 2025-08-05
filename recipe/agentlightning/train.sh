#!/bin/bash

set -e

export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
export DATA_DIR=data
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=spider
export PROJECT_NAME=AgentLightning
export TOOL_FORMAT=hermes

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train_spider.parquet \
    data.val_files=data/test_dev_500.parquet \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.n_gpus_per_node=1 \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-3B-Instruct \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.truncation='error' \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=AgentLightning \
    trainer.experiment_name=train_sql_agent \
    trainer.nnodes=1 \
    trainer.save_freq=256 \
    trainer.test_freq=32 \
    trainer.total_epochs=2 $@
