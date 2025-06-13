#!/bin/bash
# Launch script for Atropos-VERL GRPO Integration
# 
# This script launches VeRL's GRPO training with Atropos environment coordination.
# It sets up inference servers, configures environment coordination, and starts training.

# Exit on any error
set -e

# Configuration
export PYTHONPATH=$PWD:$PYTHONPATH

# Default configuration
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
ATROPOS_API_URL=${ATROPOS_API_URL:-"http://localhost:8000"}
OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/verl_atropos_checkpoints"}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_EPOCHS=${MAX_EPOCHS:-100}

# Atropos settings
WANDB_PROJECT=${WANDB_PROJECT:-"verl_atropos_grpo"}
WANDB_GROUP=${WANDB_GROUP:-"atropos_integration"}

# Advanced settings
LORA_RANK=${LORA_RANK:-0}  # Set to >0 for LoRA training
USE_FLASH_ATTN=${USE_FLASH_ATTN:-true}
INFERENCE_BACKEND=${INFERENCE_BACKEND:-"vllm"}  # or "sglang"

echo "=================================================="
echo "🚀 Atropos-VERL GRPO Training Launch"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "Atropos API: $ATROPOS_API_URL"
echo "Output Directory: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Epochs: $MAX_EPOCHS"
echo "LoRA Rank: $LORA_RANK"
echo "Inference Backend: $INFERENCE_BACKEND"
echo "=================================================="

# Check Atropos API connectivity
echo "🔍 Checking Atropos API connectivity..."
if curl -s --fail "$ATROPOS_API_URL/status" > /dev/null; then
    echo "✅ Atropos API is accessible at $ATROPOS_API_URL"
else
    echo "❌ ERROR: Cannot connect to Atropos API at $ATROPOS_API_URL"
    echo "Please ensure the Atropos server is running and accessible."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables for training
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export ATROPOS_API_URL="$ATROPOS_API_URL"

# Ray configuration
export RAY_DEDUP_LOGS=0
export RAY_DISABLE_IMPORT_WARNING=1

# Flash Attention settings
if [ "$USE_FLASH_ATTN" = "true" ]; then
    export FLASH_ATTN_FORCE_COMPILE=1
fi

echo "🔧 Environment configured, starting training..."

# Launch Atropos-VERL GRPO training
python -m verl.trainer.main_atropos \
    --config-path ../config \
    --config-name atropos_grpo_trainer \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.rollout.name="$INFERENCE_BACKEND" \
    \
    data.train_batch_size="$BATCH_SIZE" \
    \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.total_epochs="$MAX_EPOCHS" \
    trainer.output_dir="$OUTPUT_DIR" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.wandb_group="$WANDB_GROUP" \
    \
    atropos.api_url="$ATROPOS_API_URL" \
    \
    algorithm.adv_estimator="grpo_atropos" \
    \
    ray_init.num_cpus=32

echo "🎉 Training completed successfully!"
echo "Checkpoints saved to: $OUTPUT_DIR" 