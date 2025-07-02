#!/bin/bash
# Arc Vision training WITHOUT multi-turn tools (simplified for immediate start)

set -x

# Check if data exists
DATA_DIR="${HOME}/data/arc_vision/screenspot"
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: ScreenSpot data not found at ${DATA_DIR}"
    echo "Please run: python3 prepare_screenspot_data.py"
    exit 1
fi

# Launch from VERL root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../.."

# Copy config to expected location
cp "${SCRIPT_DIR}/config/arc_vision_grpo.yaml" verl/trainer/config/

# Run with TOOLS DISABLED to avoid implementation issues
python3 -m verl.trainer.main_ppo \
    --config-name arc_vision_grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    data.train_batch_size=32 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.default_local_dir=outputs/arc_vision \
    trainer.logger=['console'] \
    custom_reward_function.path=examples/arc_vision/arc_vision_custom_reward.py \
    custom_reward_function.name=arc_vision_compute_reward \
    reward_model.enable=false $@