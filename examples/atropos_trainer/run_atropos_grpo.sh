#!/bin/bash
set -e

# Configuration
export PYTHONPATH=$PWD:$PYTHONPATH
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
ATROPOS_API_URL=${ATROPOS_API_URL:-"http://localhost:8000"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/verl_atropos_checkpoints"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_EPOCHS=${MAX_EPOCHS:-100}
LORA_RANK=${LORA_RANK:-0}
INFERENCE_BACKEND=${INFERENCE_BACKEND:-"vllm"}
WANDB_PROJECT=${WANDB_PROJECT:-"verl_atropos_grpo"}

# Check API and create output dir
curl -s --fail "$ATROPOS_API_URL/status" > /dev/null || { echo "ERROR: Atropos API unreachable"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# Performance environment variables
export TOKENIZERS_PARALLELISM=true NCCL_DEBUG=WARN VLLM_LOGGING_LEVEL=WARN VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_LAUNCH_BLOCKING=0 TORCH_CUDNN_DETERMINISTIC=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export CUDA_CACHE_DISABLE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID RAY_DEDUP_LOGS=0 RAY_DISABLE_IMPORT_WARNING=1
export FLASH_ATTN_FORCE_COMPILE=1 ATROPOS_API_URL="$ATROPOS_API_URL"

# Launch training
python -m verl.trainer.main_atropos \
    --config-path ../config --config-name atropos_grpo_trainer \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.rollout.name="$INFERENCE_BACKEND" \
    data.train_batch_size="$BATCH_SIZE" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.total_epochs="$MAX_EPOCHS" \
    trainer.output_dir="$OUTPUT_DIR" \
    trainer.project_name="$WANDB_PROJECT" \
    atropos.api_url="$ATROPOS_API_URL" \
    algorithm.adv_estimator="grpo_atropos" \
    ray_init.num_cpus=32 