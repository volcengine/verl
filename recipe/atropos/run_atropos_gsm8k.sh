#!/bin/bash
# Script to run Atropos-GSM8K training

set -e

# Default values
ATROPOS_URL="${ATROPOS_API_URL:-http://localhost:9001}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "========================================"
echo "Atropos-VeRL GSM8K Training"
echo "========================================"
echo "Atropos API URL: $ATROPOS_URL"
echo "Model: $MODEL_NAME"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if Atropos is running
echo "Checking Atropos API connectivity..."
if curl -s -f -o /dev/null "$ATROPOS_URL/status"; then
    echo "✓ Atropos API is reachable"
else
    echo "✗ Cannot connect to Atropos API at $ATROPOS_URL"
    echo ""
    echo "Please ensure Atropos is running:"
    echo "1. cd /path/to/atropos"
    echo "2. python -m atroposlib.api"
    echo ""
    echo "And that GSM8K environment is registered:"
    echo "python environments/gsm8k_environment.py"
    exit 1
fi

# Export environment variables
export ATROPOS_API_URL=$ATROPOS_URL
export MODEL_NAME=$MODEL_NAME
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Change to VeRL root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../.."

# Run training
echo ""
echo "Starting training..."
python -m recipe.atropos.main_atropos_gsm8k \
    --config recipe/atropos/config/atropos_gsm8k.yaml \
    --atropos-url $ATROPOS_URL \
    $@

echo ""
echo "Training completed!"