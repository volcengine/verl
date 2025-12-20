#!/usr/bin/env bash
set -xeuo pipefail

MODEL_PATH="/path/to/Qwen2.5-0.5B-Instruct"
TRAIN_FILE="/path/to/gsm8k_train.parquet"
TEST_FILE="/path/to/gsm8k_test.parquet"

python3 -m recipe.sapo.main_sapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    trainer.project_name='verl_sapo_gsm8k' \
    trainer.experiment_name='qwen2_5_0_5b_sapo' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.save_freq=1 \
    trainer.test_freq=1
