#!/bin/bash
# Apply critical fixes to make Arc Vision training ready

echo "Applying critical fixes for Arc Vision training..."

# 1. Fix reward function name in config
echo "Fixing reward function name in config..."
sed -i.bak 's/arc_vision_compute_reward$/arc_vision_compute_score_fn/' config/arc_vision_grpo.yaml

# 2. Disable multi-turn to avoid tool issues
echo "Disabling multi-turn to avoid non-existent tool issues..."
sed -i.bak 's/enable: True/enable: False/' config/arc_vision_grpo.yaml

# 3. Fix launch script function name
echo "Fixing function name in launch script..."
sed -i.bak 's/arc_vision_compute_reward$/arc_vision_compute_score_fn/' run_arc_vision_3b_fixed.sh

# 4. Create simplified launch script without tools
cat > run_arc_vision_simple.sh << 'EOF'
#!/bin/bash
# Simplified Arc Vision training script without multi-turn tools

set -ex

# Check if data exists
DATA_DIR="${HOME}/data/arc_vision/screenspot"
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Data not found. Please run: python prepare_screenspot_data.py"
    exit 1
fi

# Launch training with simplified config
python3 -m verl.trainer.main_ppo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.adv_estimator=grpo \
    reward_model.enable=false \
    custom_reward_function.path=examples/arc_vision/arc_vision_custom_reward.py \
    custom_reward_function.name=arc_vision_compute_score_fn \
    trainer.n_gpus_per_node=2 \
    trainer.default_local_dir=outputs/arc_vision \
    trainer.logger=['console'] \
    trainer.total_epochs=5 $@
EOF

chmod +x run_arc_vision_simple.sh

echo "Fixes applied! You can now run:"
echo "  bash run_arc_vision_simple.sh"
echo ""
echo "Or if you prefer the full config:"
echo "  bash run_arc_vision_grpo.sh"