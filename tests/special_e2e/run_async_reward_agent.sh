#!/usr/bin/env bash
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# The below code in this distribution has been modified by Tencent ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) Tencent.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xeuo pipefail

# Test script for async_reward_agent E2E regression testing
# This script runs async_reward_agent with both FSDP2 and Megatron backends

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}  # fsdp2 or megatron

# Download model if not exists
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

PROJECT_DIR="$(pwd)"
REWARD_FILE="$PROJECT_DIR/recipe/async_reward_agent/reward/gsm8k.py"

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

# Response length parameters
max_prompt_length=512
max_response_length=512

# Training parameters
loss_agg_mode="token-mean"
train_prompt_bsz=64
n_resp_per_prompt=5
train_prompt_mini_bsz=16

# Temperature parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

n_gpus_training=${NUM_GPUS}

exp_name="$(basename "${MODEL_ID,,}")-async-reward-agent-${ACTOR_STRATEGY}-minimal"

echo "Running async_reward_agent with ${ACTOR_STRATEGY} strategy"

# Common parameters for both FSDP2 and Megatron
common_params=(
    data.train_files="${HOME}/data/gsm8k/train.parquet"
    data.val_files="${HOME}/data/gsm8k/test.parquet"
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True 
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 
    trainer.logger=['console']
    trainer.project_name='verl-test'
    trainer.experiment_name="${exp_name}"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.total_epochs=2
    trainer.total_training_steps=2
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${n_gpus_training}
    custom_reward_function.path=${REWARD_FILE} \
    reward_model.reward_manager=batch \
    reward_model.launch_reward_fn_async=True \
    +mini_batch_pipeline=True
)

if [ "${ACTOR_STRATEGY}" == "fsdp2" ]; then
    echo "Running with FSDP2 strategy..."
    # FSDP2 specific parameters
    gen_tp=2
    ref_offload=True
    actor_offload=False

    python3 -m recipe.async_reward_agent.main_ppo \
        --config-path="${PROJECT_DIR}/verl/trainer/config" \
        "${common_params[@]}" \
        actor_rollout_ref.actor.strategy=fsdp2 \
        critic.strategy=fsdp2 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        custom_reward_function.name=Gsm8kAgent \
        actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} $@

elif [ "${ACTOR_STRATEGY}" == "megatron" ]; then
    echo "Running with Megatron strategy..."
    # Megatron specific parameters
    gen_tp=2
    train_tp=2
    train_pp=1
    ref_offload=True
    actor_offload=True

    python3 -m recipe.async_reward_agent.main_ppo \
        --config-path="${PROJECT_DIR}/verl/trainer/config" \
        --config-name='ppo_megatron_trainer.yaml'\
        "${common_params[@]}" \
        actor_rollout_ref.actor.strategy=megatron \
        critic.strategy=megatron \
        actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
        custom_reward_function.name=compute_score_per_sample \
        actor_rollout_ref.ref.megatron.param_offload=${ref_offload} $@
else
    echo "Error: Unknown strategy ${ACTOR_STRATEGY}. Please use 'fsdp2' or 'megatron'"
    exit 1
fi

echo "Async reward agent E2E test completed successfully with ${ACTOR_STRATEGY} strategy"
