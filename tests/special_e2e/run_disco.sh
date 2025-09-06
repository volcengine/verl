#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

exp_name="$(basename "${MODEL_ID,,}")-disco-minimal"

nnodes=1
n_gpus_per_node=4
ppo_micro_batch_size_per_gpu=4
rollout_n=8

loss_mode='disco'
### score function selection for disco
score_func='logL'  # Options: 'logL', 'Lratio'
tau=10  ### tau=10 is recommended for 'logL',  tau=1 is recommended for 'Lratio'

# Train over a single node, 4 A100-80GB GPUs.
python3 -m recipe.disco.main_disco \
    algorithm.adv_estimator=disco \
    algorithm.filter_groups.enable=False \
    data.train_files=./recipe/disco/data/deepscaler_preview.parquet \
    data.val_files=./recipe/disco/data/aime24.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=36864 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    +actor_rollout_ref.ref.enable=False  \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode  \
    actor_rollout_ref.actor.policy_loss.score_func=$score_func \
    actor_rollout_ref.actor.policy_loss.delta=1e-4 \
    actor_rollout_ref.actor.policy_loss.beta=1e3 \
    actor_rollout_ref.actor.policy_loss.tau=$tau \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl-test' \
    trainer.experiment_name=$exp_name \
    trainer.balance_batch=False  \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}" \
    trainer.total_training_steps=1 \
    trainer.resume_mode=auto
