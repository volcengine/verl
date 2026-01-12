#!/bin/bash
set -x


# Determine how many nodes were allocated. 
NNODES=1

# wandb logging
backend=megatron # fsdp, fsdp2, megatron
project_name=GDPO
experiment_name=qwen-1.5b-GDPO-$backend

# ===================================== Algorithm =====================================
adv_estimator=gdpo

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

actor_lr=1e-6
critic_warmup=0

# ===================================== Data/Model =====================================

default_local_dir="$HOME/results"
actor_model_path="Qwen/Qwen2.5-1.5B-Instruct"

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

critic_model_path=$actor_model_path

max_prompt_length=$((1024))
max_response_length=$((1024))

train_batch_size=256
ppo_mini_batch_size=128
n_resp_per_prompt=8
n_resp_per_prompt_val=1

# ===================================== Training =====================================
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 6))

enable_gradient_checkpointing=True
param_offload=False
optimizer_offload=False


VAL_BEFORE_TRAIN=False
SAVE_FREQ=30
TEST_FREQ=10
TOTAL_EPOCHS=100

# FSDP parallelism config
usp_size=1
tp_size=1
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$usp_size" \
    actor_rollout_ref.actor.tensor_model_parallel_size=$tp_size

# Megatron parallelism config
train_tp=2
train_cp=1
train_pp=1
train_vpp=null

ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$train_tp \
    actor_rollout_ref.actor.megatron.context_parallel_size=$train_cp \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$train_pp \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$train_vpp \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

# Critic model config
CIRITC_CONFIG="
    critic.model.path=$critic_model_path \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$usp_size"

CRITIC_FSDP_CONFIG="${ACTOR_FSDP_CONFIG//actor_rollout_ref.actor/critic.model}"
CRITIC_MEGATRON_CONFIG="${ACTOR_MEGATRON_CONFIG//actor_rollout_ref.actor/critic}"

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_MEGATRON_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
else # fsdp, fsdp2
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
    if [[ $adv_estimator == "gae" ]]; then
        CIRITC_CONFIG="$CIRITC_CONFIG $CRITIC_FSDP_CONFIG"
    else
        CIRITC_CONFIG=""
    fi
fi

# ===================================== Inference =====================================
rollout_engine=vllm
infer_tp=2
infer_dp=1
gpu_memory_utilization=0.8

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_engine \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Reward =====================================
REWARD_CONFIG="
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}"


export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ================================= Launch Training ================================
cd /workspace/wanghc6@xiaopeng.com/train/verl

python -m verl.trainer.main_ppo \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    data.train_files="$gsm8k_train_path" \
    data.val_files="$gsm8k_test_path" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=$SLURM_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.log_val_generations=100 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    $ACTOR_CONFIG \
    $CIRITC_CONFIG \
    $ROLLOUT_CONFIG \
    $REWARD_CONFIG

