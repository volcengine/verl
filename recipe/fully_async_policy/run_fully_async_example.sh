#!/bin/bash
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

set -x

# 实验配置
project_name='FullyAsyncPPO'
exp_name='async-qwen2.5-7b-test'

# 模型和数据路径
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"~/data/train.parquet"}
VAL_FILE=${VAL_FILE:-"~/data/val.parquet"}

# 硬件配置
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# 异步训练资源分配
n_gpus_rollout=3  # rollout专用GPU数量
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))  # 训练GPU数量

echo "==================================="
echo "完全异步PPO训练启动"
echo "==================================="
echo "模型路径: $MODEL_PATH"
echo "训练数据: $TRAIN_FILE"
echo "验证数据: $VAL_FILE"
echo "节点数: $NNODES"
echo "每节点GPU数: $NGPUS_PER_NODE"
echo "Rollout GPU数: $n_gpus_rollout"
echo "训练GPU数: $n_gpus_training"
echo "==================================="

# 算法参数
temperature=1.0
top_p=1.0
top_k=-1

# 序列长度
max_prompt_length=1024
max_response_length=1024

# 异步训练参数
staleness_threshold=3
max_staleness_allowed=5
max_queue_size=1000
min_batch_count=1
batch_timeout=30.0

# 训练参数
train_batch_size=128
total_training_steps=1000
save_freq=100
val_freq=50

# 设置环境变量
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VERL_LOGGING_LEVEL=INFO

# 启动训练
python -m recipe.one_step_off_policy.fully_async_main \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$exp_name" \
    trainer.device=cuda \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    \
    # 模型配置
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    \
    # Rollout配置
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n_gpus=$n_gpus_rollout \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    \
    # Actor配置
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    \
    # Critic配置
    critic.model.path="$MODEL_PATH" \
    critic.optim.lr=1e-5 \
    critic.fsdp_config.param_offload=false \
    \
    # 异步训练配置
    async_training.staleness_threshold=$staleness_threshold \
    async_training.max_staleness_allowed=$max_staleness_allowed \
    async_training.max_queue_size=$max_queue_size \
    async_training.min_batch_count=$min_batch_count \
    async_training.batch_timeout=$batch_timeout \
    \
    # 训练配置
    trainer.total_training_steps=$total_training_steps \
    trainer.save_freq=$save_freq \
    trainer.val_freq=$val_freq \
    trainer.critic_warmup=0 \
    \
    # 算法配置
    algorithm.adv_estimator=gae \
    algorithm.cliprange=0.2 \
    algorithm.vf_coeff=0.1 \
    algorithm.entropy_coeff=0.01 \
    algorithm.kl_coeff=0.1 \
    \
    # 日志配置
    trainer.logger='["console", "wandb"]' \
    trainer.val_before_train=false

echo "==================================="
echo "完全异步PPO训练完成"
echo "==================================="

