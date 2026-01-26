#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# 一、环境设置和常用参数设置
# ============================================================================

## 基础环境设置
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
# TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
export TASK_QUEUE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export CPU_AFFINITY_CONF=1
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
ulimit -n 32768

# Project Configuration
project_name='GSPO'
exp_name='GSPO-Qwen3-32B'

# Node Info
NNODES=${NNODES:-4}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

## 路径配置
# Model Weights Paths
MODEL_PATH=./ckpt/Qwen3-32B
CKPTS_DIR=./ckpt/Qwen3-32B-save

# File System Paths
TRAIN_FILE=./data/dapo-math-17k.parquet
TEST_FILE=./data/dapo-math-17k.parquet

# Ray Configuration
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

## 数据长度和批次参数设置
# Data Length Configuration
max_prompt_length=$((1024 * 16))
max_response_length=$((1024 * 16))

# Training Batch Configuration
train_prompt_bsz=256
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=64
n_resp_per_prompt=16

## 算法核心参数设置
# Algorithm Configuration
adv_estimator=grpo
loss_mode=gspo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# GSPO Loss Configuration
clip_ratio_low=0.0003
clip_ratio_high=0.0004
loss_agg_mode="seq-mean-token-mean"

# Generation Parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

## 性能和内存管理参数设置
# Performance and Memory Management Configuration
offload=True
use_dynamic_bsz=True
sp_size=4
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

## FSDP训练后端参数配置
# FSDP Parallelism Configuration
actor_strategy=fsdp2
ref_strategy=fsdp2
fsdp_size=-1

## vLLM推理后端参数配置
# vLLM Generation Configuration
gen_tp=4
gpu_memory_utilization=0.90
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ============================================================================
# 二、模块化配置
# ============================================================================

## 数据集配置
DATA_CONFIG=(
    # File Paths
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    # Data Structure
    data.prompt_key=prompt
    # Batch and Length Configuration
    data.train_batch_size=${train_prompt_bsz}
    +data.gen_batch_size=${gen_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    # Preprocessing
    data.truncation='left'
)

## 模型配置
MODEL_CONFIG=(
    # Model Path
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # Model Processing
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

## 算法配置
ALGORITHM_CONFIG=(
    # Advantage Estimation
    algorithm.adv_estimator=${adv_estimator}
    # KL Divergence Control
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

## Actor策略配置
ACTOR_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.strategy=${actor_strategy}
    # Loss Function Configuration
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    # FSDP Parallelism Strategy
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    # Memory Optimization
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    # Entropy Computation Optimization
    actor_rollout_ref.actor.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
)

## REF策略配置
REF_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.strategy=${ref_strategy}
    # Log Probability Inference
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # FSDP Parallelism Strategy
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}
    # Memory Optimization
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload}
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
    # Entropy Computation Optimization
    actor_rollout_ref.ref.entropy_checkpointing=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

## Rollout生成配置
ROLLOUT_CONFIG=(
    # Rollout Engine
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.calculate_log_probs=True
    # Generation Parameters
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k="${top_k}"
    # Log Probability Inference
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    # Memory Management
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    # Parallelism Strategy
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    # Performance Optimization
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    # CudaGraph Configuration
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[8, 16, 32, 64, 128, 192, 256, 384]"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    # Validation Generation
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
)

## 训练器配置
TRAINER_CONFIG=(
    # Logger Configuration
    trainer.logger='["console"]'
    # Project Settings
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    # Hardware Configuration
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    # Training Schedule
    trainer.total_epochs=10
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=100
    # Checkpoint Directory
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.balance_batch=True
)

# ============================================================================
# 三、启动命令
# ============================================================================

# Main GSPO Training Command
python3 -m verl.trainer.main_ppo \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@" 2>&1 | tee "logs/verl_qwen3_32b_$(date +%Y%m%d_%H%M).log"