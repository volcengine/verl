#!/usr/bin/env bash
set -Eeuo pipefail

# Train Qwen 7B with mixed DP+SP on 8×A100 (40GB), 1 epoch GSM8K PPO.
# Actor: DeepSpeed ZeRO-2 + Ulysses SP=2 (=> DP≈4 over 8 GPUs)
# Rollout: vLLM TP=2, util=0.2
# Seeds fixed; robust quoting; logs to $HOME.

ROOT_DIR_DEFAULT="/home/ubuntu/verl"
export ROOT_DIR="${ROOT_DIR:-$ROOT_DIR_DEFAULT}"
export PPO_DATA_DIR="${PPO_DATA_DIR:-$HOME/data/gsm8k_ppo}"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Safer env for stability
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

CONFIG_PATH="$ROOT_DIR/examples/benchmarks/fsdp_vs_deepspeed_ppo/config"
LOG_FILE="$HOME/qwen7b_sp2_dp4_epoch1_$(date +'%Y%m%d_%H%M%S').log"

echo "--- Qwen7B SP=2 + DP=4 (8 GPUs), vLLM TP=2, util=0.2 | logging: $LOG_FILE ---"

ARGS=(
  --config-path "$CONFIG_PATH"
  --config-name deepspeed_ppo_benchmark

  # Data
  "data.train_files=[\"$PPO_DATA_DIR/train.parquet\"]"
  "data.val_files=[\"$PPO_DATA_DIR/test.parquet\"]"
  data.train_batch_size=128
  data.seed=42

  # Model override to 7B
  actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # DeepSpeed Actor (SP=2 -> DP=8/2=4)
  # 使用 ZeRO-2（可选开启优化器 CPU offload 进一步降压）
  actor_rollout_ref.actor.zero_stage=2
  actor_rollout_ref.actor.ppo_mini_batch_size=128
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.actor.gradient_accumulation_steps=8
  actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16
  actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=2
  # 仅优化器 offload 即可，大幅降低峰值显存；如仍吃紧可再开 param_offload
  actor_rollout_ref.actor.deepspeed_config.param_offload=False
  actor_rollout_ref.actor.deepspeed_config.optimizer_offload=True

  # Critic (match actor scale; can relax micro-batch if needed)
  critic.zero_stage=2
  critic.ppo_mini_batch_size=128
  critic.ppo_micro_batch_size_per_gpu=4
  critic.gradient_accumulation_steps=8
  critic.deepspeed_config.model_dtype=bf16
  critic.deepspeed_config.mixed_precision=bf16
  critic.deepspeed_config.ulysses_sequence_parallel_size=2
  critic.deepspeed_config.param_offload=False
  critic.deepspeed_config.optimizer_offload=True
  critic.model.path=Qwen/Qwen2.5-7B-Instruct

  # vLLM rollout
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.15
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.seed=42

  # Trainer / Ray
  trainer.total_epochs=1
  trainer.total_training_steps=null
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  "trainer.logger=[\"console\"]"
  trainer.resume_mode=disable
  +ray_kwargs.ray_init.num_gpus=8
  +ray_kwargs.ray_init.num_cpus=16
)

python3 -m verl.trainer.main_ppo "${ARGS[@]}" |& tee "$LOG_FILE"

echo "--- Done. Log: $LOG_FILE ---"
