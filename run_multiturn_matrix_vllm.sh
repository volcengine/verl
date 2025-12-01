#!/usr/bin/env bash

# Run a small matrix of multiturn eval commands and emit per-case logs.
# Each case writes its stdout/stderr to logs_multiturn/<case>.log so you
# can quickly spot which combinations fail.

set -u

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/verl/trainer/config"
LOG_DIR="$PROJECT_DIR/logs_multiturn"

mkdir -p "$LOG_DIR"

TEXT_DATA="${TEXT_DATA:-/home/gty/minhoucheng/data/gsm8k_w_tool/test.parquet}"
TEXT_CHECKPOINT="${TEXT_CHECKPOINT:-/home/gty/minhoucheng/checkpoint/vllm_multiturn_async_gsm8k_code_interpreter_n8}"
TEXT_TOOL_CONFIG="${TEXT_TOOL_CONFIG:-$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml}"
TEXT_MODEL="${TEXT_MODEL:-/home/gty/minhoucheng/Qwen/Qwen3-1.7B}"

MM_DATA="${MM_DATA:-/home/gty/minhoucheng/data/geo3k_multiturn_w_tool/test.parquet}"
MM_CHECKPOINT="${MM_CHECKPOINT:-/home/gty/minhoucheng/checkpoint/geo3k_async_rl_4b_thinking}"
MM_TOOL_CONFIG="${MM_TOOL_CONFIG:-$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml}"
MM_MODEL="${MM_MODEL:-/home/gty/minhoucheng/Qwen/Qwen3-VL-4B-Thinking}"

run_case() {
  local name="$1" mode="$2" engine="$3" tp="$4" modality="$5"
  local logfile="$LOG_DIR/${name}.log"

  echo "=== Running $name (mode=$mode, engine=$engine, tp=$tp, modality=$modality) ==="

  (
    set -x
    ulimit -n 65535 || true

  if [[ "$modality" == "mm" ]]; then
    model_path="$MM_MODEL"
    data_file="$MM_DATA"
    tool_config="$MM_TOOL_CONFIG"
    ckpt_dir="$MM_CHECKPOINT"
  else
    model_path="$TEXT_MODEL"
    data_file="$TEXT_DATA"
    tool_config="$TEXT_TOOL_CONFIG"
    ckpt_dir="$TEXT_CHECKPOINT"
  fi

    python3 -m verl.trainer.main_multiturn_eval \
      --config-path="$CONFIG_PATH" \
      --config-name='multiturn_eval' \
      data.eval_files="$data_file" \
      data.eval_batch_size=64 \
      data.max_prompt_length=2048 \
      data.max_response_length=2048 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.return_raw_chat=True \
      data.need_tools_kwargs=True \
      actor_rollout_ref.model.path="$model_path" \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.rollout.tensor_model_parallel_size="$tp" \
      actor_rollout_ref.rollout.name="$engine" \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
      actor_rollout_ref.rollout.mode="$mode" \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.rollout.multi_turn.tool_config_path="$tool_config" \
      actor_rollout_ref.rollout.multi_turn.interaction_config_path=null \
      actor_rollout_ref.rollout.agent.num_workers=8 \
      trainer.n_gpus_per_node=2 \
      trainer.nnodes=1 \
      reward_model.enable=False \
      reward_model.enable_resource_pool=False \
      reward_model.reward_manager=dapo \
      +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
      +reward_model.reward_kwargs.overlong_buffer_cfg.len=256 \
      +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1 \
      +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
      +reward_model.reward_kwargs.max_resp_len=2048 \
      checkpoint_dir="$ckpt_dir" \
      evaluation.max_batches=2 \
      evaluation.max_samples=null \
      output.path=./eval_results \
      output.scores_path=evaluation_scores.json
  ) >"$logfile" 2>&1

  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "[OK]  $name -> $logfile"
  else
    echo "[FAIL] $name (exit $status) -> $logfile"
  fi
}

# Text cases
run_case "text_async_vllm_tp1"  "async" "vllm"   1 "text"
run_case "text_async_vllm_tp2"  "async" "vllm"   2 "text"

# Multimodal cases
run_case "mm_async_vllm_tp1"  "async" "vllm"   1 "mm"
run_case "mm_async_vllm_tp2"  "async" "vllm"   2 "mm"


echo "All cases submitted. Check logs in $LOG_DIR"
