#!/bin/bash
set -x

echo "================================================"
echo "Running DeepSpeed PPO Benchmark"
echo "================================================"

# Navigate to verl root
cd /home/paperspace/verl

# Check if data exists
if [ ! -f "$HOME/data/gsm8k/train.parquet" ]; then
    echo "Error: GSM8K data not found. Please run prepare_data.sh first."
    exit 1
fi

# Create results directory
mkdir -p ./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/paperspace/verl:$PYTHONPATH
export PYTHONHASHSEED=42
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# Log file
LOG_FILE=./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed/training.log

python3 -m verl.trainer.main_ppo \
    --config-path /home/paperspace/verl/examples/benchmarks/fsdp_vs_deepspeed_ppo/config \
    --config-name deepspeed_ppo_benchmark \
    data.seed=42 \
    trainer.default_local_dir=./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed \
    $@ 2>&1 | tee "$LOG_FILE"

echo "================================================"
echo "DeepSpeed PPO Benchmark Complete!"
echo "Results saved to: ./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed"
echo "Log saved to: $LOG_FILE"
echo "================================================"
