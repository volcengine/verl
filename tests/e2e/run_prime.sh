#!/usr/bin/env bash
set -x

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

train_prompt_bsz=16 # 8n
train_prompt_mini_bsz=$((train_prompt_bsz / 2)) # 4n
n_resp_per_prompt=4
train_traj_mini_bsz=$((train_prompt_mini_bsz * n_resp_per_prompt)) # 16n
train_traj_micro_bsz=$((train_traj_mini_bsz / 2)) # 8n
num_gpus=8
train_traj_micro_bsz_per_gpu=$((train_traj_micro_bsz / num_gpus)) # n

exp_name="$(basename "${MODEL_ID,,}")-prime-minimal-$(git rev-parse --short HEAD)"

python3 -m recipe.prime.main_prime \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    data.oversample_factor=4 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_ID}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.model.path="${MODEL_ID}" \
    reward_model.micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    reward_model.model.update=before \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=${train_prompt_bsz} \
    reward_model.reward_manager=prime \
    trainer.val_before_train=False \
    trainer.logger=['console'] \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=2 $@
