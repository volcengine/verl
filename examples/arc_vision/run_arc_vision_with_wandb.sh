#!/bin/bash
# Arc Vision RL training with optional WandB support

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

# Check if WandB API key is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "WandB API key not found. Using console logging only."
    echo "To enable WandB, run: export WANDB_API_KEY=your_api_key"
    LOGGER="['console']"
else
    echo "WandB API key found. Enabling WandB logging."
    LOGGER="['console','wandb']"
fi

# Launch Arc Vision RL training from VERL root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.."  # Go to VERL root

# Copy our config to the expected location
cp "${SCRIPT_DIR}/config/arc_vision_grpo.yaml" verl/trainer/config/

# Launch training with dynamic logger configuration
python3 -m verl.trainer.main_ppo \
    --config-name arc_vision_grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    actor_rollout_ref.rollout.name=$ENGINE \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=outputs/arc_vision \
    trainer.logger=$LOGGER \
    custom_reward_function.path=examples/arc_vision/arc_vision_custom_reward.py \
    custom_reward_function.name=arc_vision_compute_reward $@