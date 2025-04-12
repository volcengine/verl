#!/usr/bin/env bash
set -x

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL_ID}}

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

RM_PAD=${RM_PAD:-True}
SP_SIZE=${SP_SIZE:-1}
SEQ_BALANCE=${SEQ_BALANCE:-False}
LIGER=${LIGER:-False}

huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_liger="${LIGER}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz="${SEQ_BALANCE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.ulysses_sequence_parallel_size="${SP_SIZE}" \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.use_dynamic_bsz="${SEQ_BALANCE}" \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.ulysses_sequence_parallel_size="${SP_SIZE}" \
    reward_model.model.path="${MODEL_PATH}" \
    reward_model.model.use_remove_padding="${RM_PAD}" \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.use_dynamic_bsz="${SEQ_BALANCE}" \
    reward_model.forward_max_token_len_per_gpu=8192 \
    reward_model.micro_batch_size_per_gpu=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name='qwen_e2e_ci_hybrid_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=1 $@
