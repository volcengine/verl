#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

# ============================================================================
# Memory-Optimized PPO Training Example
# ============================================================================
# This script demonstrates all memory optimization features in verl for
# training large models with limited GPU memory.
#
# Memory optimization features enabled:
# 1. Activation Offloading - Offload activations to CPU during forward pass
# 2. Gradient Checkpointing - Recompute activations during backward pass
# 3. Parameter/Optimizer Offloading - Offload model states to CPU
# 4. reshard_after_forward - Full sharding strategy for FSDP
# 5. Dynamic batch sizing - Efficient memory utilization with variable lengths
# 6. Reduced gpu_memory_utilization - Leave room for training memory
#
# For CPU memory efficiency, set LD_PRELOAD to use tcmalloc or jemalloc:
#   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
#
# Reference: https://github.com/volcengine/verl/issues/144
# ============================================================================

set -x

# Data paths - adjust to your data location
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

# Model path - adjust to your model
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-7B-Instruct"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    `# ============ Actor Model Configuration ============` \
    actor_rollout_ref.model.path=$MODEL_PATH \
    \
    `# Memory Optimization 1: Gradient Checkpointing` \
    `# Trades compute for memory by recomputing activations during backward` \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# Memory Optimization 2: Activation Offloading` \
    `# Offloads activations to CPU during forward pass, reloads during backward` \
    `# Works with FSDP/FSDP2 only. Combines with gradient checkpointing.` \
    actor_rollout_ref.model.enable_activation_offload=True \
    \
    `# Memory Optimization 3: Remove Padding` \
    `# More efficient computation by removing padding tokens` \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    `# ============ Actor Training Configuration ============` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    \
    `# Memory Optimization 4: Dynamic Batch Sizing` \
    `# Efficiently packs sequences to maximize GPU utilization` \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    \
    `# Memory Optimization 5: FSDP Configuration` \
    `# reshard_after_forward=True enables ZeRO-3 style full sharding` \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    \
    `# Memory Optimization 6: Parameter/Optimizer Offloading` \
    `# Offload parameters and optimizer states to CPU` \
    `# Set to True for very large models (increases training time)` \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    `# ============ Reference Model Configuration ============` \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    \
    `# ============ Rollout Configuration ============` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    `# Memory Optimization 7: Inference Memory Utilization` \
    `# Lower value leaves more memory for training` \
    `# When using param_offload, can increase to 0.8-0.9` \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    \
    `# Memory Optimization 8: Disable CUDA Graphs` \
    `# CUDA graphs use additional memory that cannot be offloaded` \
    actor_rollout_ref.rollout.enforce_eager=True \
    \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    \
    `# ============ Critic Model Configuration ============` \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.enable_activation_offload=True \
    critic.model.use_remove_padding=True \
    \
    critic.optim.lr=1e-5 \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=16384 \
    \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.model.fsdp_config.reshard_after_forward=True \
    \
    `# ============ Algorithm Configuration ============` \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    `# ============ Trainer Configuration ============` \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_memory_optimized' \
    trainer.experiment_name='memory_optimized_ppo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    "$@"
