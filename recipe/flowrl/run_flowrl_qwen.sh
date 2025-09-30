#!/bin/bash
# FlowRL training script for Qwen models

set -e

# Default configurations
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
DATA_PATH=${DATA_PATH:-"path/to/your/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./flowrl_outputs"}
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-256}

# Export CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting FlowRL training..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"

# Run FlowRL training
python3 recipe/flowrl/main_flowrl.py \
    --config recipe/flowrl/config/flowrl_qwen.yaml \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.save_dir=$OUTPUT_DIR \
    data.train_batch_size=$BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    data.train_files=[$DATA_PATH] \
    "$@"

echo "FlowRL training completed!"
echo "Results saved to: $OUTPUT_DIR"