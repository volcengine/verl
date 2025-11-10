#!/usr/bin/env bash
set -euo pipefail

# 2-GPU Data Parallel (DP=2, TP=1), 2 epochs (~116 steps)

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT_DIR"

PPO_DATA_DIR="$HOME/data/gsm8k_ppo"
OUT_DIR_REL="examples/benchmarks/fsdp_vs_deepspeed_ppo/results"
OUT_DIR_ABS="$ROOT_DIR/$OUT_DIR_REL"
mkdir -p "$OUT_DIR_ABS"

if [ ! -f "$PPO_DATA_DIR/train.parquet" ] || [ ! -f "$PPO_DATA_DIR/test.parquet" ]; then
  echo "[prepare] preprocessing GSM8K PPO data to $PPO_DATA_DIR ..."
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "$PPO_DATA_DIR"
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONHASHSEED=42
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

TRAIN_FILES="['$PPO_DATA_DIR/train.parquet']"
VAL_FILES="['$PPO_DATA_DIR/test.parquet']"

LOG_PATH="$OUT_DIR_ABS/dp_2gpu_ep2.log"

echo "[run] DP=2 (2 GPUs), TP=1, 2 epochs"
python3 -m verl.trainer.main_ppo \
  --config-path "$ROOT_DIR/examples/benchmarks/fsdp_vs_deepspeed_ppo/config" \
  --config-name deepspeed_ppo_benchmark \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.seed=42 \
  data.train_batch_size=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.gradient_accumulation_steps=8 \
  actor_rollout_ref.actor.zero_stage=0 \
  actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16 \
  actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.12 \
  critic.ppo_mini_batch_size=128 \
  critic.ppo_micro_batch_size_per_gpu=8 \
  critic.gradient_accumulation_steps=8 \
  critic.deepspeed_config.model_dtype=bf16 \
  critic.deepspeed_config.mixed_precision=bf16 \
  trainer.total_training_steps=116 \
  trainer.total_epochs=2 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.resume_mode=disable \
  +ray_kwargs.ray_init.num_gpus=2 \
  +ray_kwargs.ray_init.num_cpus=8 \
  actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size=1 \
  > "$LOG_PATH" 2>&1

echo "[done] Log at: $LOG_PATH"

