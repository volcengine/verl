#!/usr/bin/env bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

model_id=Qwen/Qwen2.5-0.5B
model_path=${HOME}/models/Qwen/${model_id}

# n >= 2
train_prompt_bsz=4 # 2n >= 4
train_prompt_mini_bsz=$((train_prompt_bsz / 2)) # n
n_resp_per_prompt=4
train_traj_mini_bsz=$((train_prompt_mini_bsz * n_resp_per_prompt)) # 4n
train_traj_micro_bsz=$((train_traj_mini_bsz / 2)) # 2n
num_gpus=2
train_traj_micro_bsz_per_gpu=$((train_traj_micro_bsz / num_gpus)) # n

huggingface-cli download ${model_id} --local-dir "${model_path}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${HOME}"/data/gsm8k/train.parquet \
    data.val_files="${HOME}"/data/gsm8k/test.parquet \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${num_gpus} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2p5_0p5b_function_rm' \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.total_training_steps=2 $@