#!/usr/bin/env bash
set -Eeuo pipefail

# 快速验证：4-GPU，SP=2，DP=2，ZeRO-2（8 steps），释放 vLLM 显存，避免 OOM

ROOT_DIR_DEFAULT="/home/ubuntu/verl"
export ROOT_DIR="${ROOT_DIR:-$ROOT_DIR_DEFAULT}"
export PPO_DATA_DIR="${PPO_DATA_DIR:-$HOME/data/gsm8k_ppo}"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

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
LOG_FILE="$HOME/validate_sp2_dp2_zero2_$(date +'%Y%m%d_%H%M%S').log"

echo "--- 验证运行：SP=2, DP=2, ZeRO-2, steps=8 | 日志: $LOG_FILE ---"

ARGS=(
  --config-path "$CONFIG_PATH"
  --config-name deepspeed_ppo_benchmark
  "data.train_files=[\"$PPO_DATA_DIR/train.parquet\"]"
  "data.val_files=[\"$PPO_DATA_DIR/test.parquet\"]"
  data.train_batch_size=128

  actor_rollout_ref.actor.zero_stage=2
  actor_rollout_ref.actor.ppo_mini_batch_size=128
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.actor.gradient_accumulation_steps=8
  actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16
  actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=2

  critic.zero_stage=2
  critic.ppo_mini_batch_size=128
  critic.ppo_micro_batch_size_per_gpu=4
  critic.gradient_accumulation_steps=8
  critic.deepspeed_config.model_dtype=bf16
  critic.deepspeed_config.mixed_precision=bf16
  critic.deepspeed_config.ulysses_sequence_parallel_size=2
  critic.model.enable_gradient_checkpointing=True

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.15
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.dtype=bfloat16

  trainer.total_training_steps=8
  trainer.total_epochs=1
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  "trainer.logger=[\"console\"]"
  trainer.resume_mode=disable
  +ray_kwargs.ray_init.num_gpus=4
  +ray_kwargs.ray_init.num_cpus=8
)

python3 -m verl.trainer.main_ppo "${ARGS[@]}" |& tee "$LOG_FILE"

echo "--- 验证结束，日志: $LOG_FILE ---"

