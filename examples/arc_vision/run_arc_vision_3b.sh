#!/bin/bash
# Arc Vision RL training script for Qwen2.5-VL-3B with confidence-gated tool learning

set -x

# Check if data exists
DATA_DIR="${HOME}/data/arc_vision/screenspot"
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: ScreenSpot data not found at ${DATA_DIR}"
    echo "Please run: cd examples/arc_vision && python prepare_screenspot_data.py"
    exit 1
fi

# Set engine (default to sglang for multi-turn support)
ENGINE=${1:-sglang}

# Launch Arc Vision RL training from VERL root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.."  # Go to VERL root

# Use our YAML config with proper Hydra syntax
python3 -m verl.trainer.main_ppo \
    --config-path examples/arc_vision/config \
    --config-name arc_vision_grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    actor_rollout_ref.rollout.name=$ENGINE \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=outputs/arc_vision $@