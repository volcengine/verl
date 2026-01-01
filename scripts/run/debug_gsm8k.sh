#!/bin/bash
# set -x

eval "$(conda shell.bash hook)"
conda activate verl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="$(pwd)"
SAVE_PATH="${PROJECT_DIR}/data"
export HF_HOME=$SAVE_PATH
export NCCL_P2P_DISABLE=1

export VLLM_CACHE_ROOT=$SAVE_PATH/vllm_cache
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN # vllm + qwen2-7b with flash_attn has some issues

# export FLASHINFER_BASE_DIR=$SAVE_PATH/flashinfer_cache
# export VLLM_USE_FLASHINFER=0

# Ray will kill the most recent task if node memory exceeds this threshold (default 0.95).
# Bump it slightly to avoid spurious kills when the node is already busy with other jobs.
export RAY_memory_usage_threshold=0.98

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.dataloader_num_workers=0 \
    data.return_full_prompt=True \
    data.train_files=$SAVE_PATH/gsm8k/train.parquet \
    data.val_files=$SAVE_PATH/gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=10 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_sft' \
    trainer.experiment_name='NEW/ppo_micro_batch_size_per_gpu2' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.resume_mode="disable" \
    trainer.total_epochs=15 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.sft.enabled=True
