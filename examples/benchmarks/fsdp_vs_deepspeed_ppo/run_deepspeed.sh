#!/bin/bash
set -x

echo "================================================"
echo "Running DeepSpeed PPO Benchmark"
echo "================================================"

# Navigate to verl root
cd /home/paperspace/verl

# Preprocess GSM8K data for PPO training
echo "================================================"
echo "Step 1: Checking and preprocessing GSM8K data"
echo "================================================"

PPO_DATA_DIR="$HOME/data/gsm8k_ppo"

if [ ! -f "$PPO_DATA_DIR/train.parquet" ] || [ ! -f "$PPO_DATA_DIR/test.parquet" ]; then
    echo "PPO-formatted data not found. Running preprocessing..."

    # Run preprocessing script
    python3 examples/data_preprocess/gsm8k.py \
        --local_save_dir "$PPO_DATA_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ Data preprocessing completed successfully"
    else
        echo "✗ Error: Data preprocessing failed"
        exit 1
    fi
else
    echo "✓ PPO-formatted data already exists at $PPO_DATA_DIR"
fi

# Verify preprocessed data
if [ ! -f "$PPO_DATA_DIR/train.parquet" ]; then
    echo "Error: Preprocessed training data not found at $PPO_DATA_DIR/train.parquet"
    exit 1
fi

echo ""
echo "================================================"
echo "Step 2: Running DeepSpeed PPO Training"
echo "================================================"

# Create results directory
mkdir -p ./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/paperspace/verl:$PYTHONPATH
export PYTHONHASHSEED=42
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# Prepare data file paths
train_files="['$PPO_DATA_DIR/train.parquet']"
test_files="['$PPO_DATA_DIR/test.parquet']"

# Log file
LOG_FILE=./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed/training.log

python3 -m verl.trainer.main_ppo \
    --config-path /home/paperspace/verl/examples/benchmarks/fsdp_vs_deepspeed_ppo/config \
    --config-name deepspeed_ppo_benchmark \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.seed=42 \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.deepspeed_config.param_offload=False \
    actor_rollout_ref.actor.deepspeed_config.optimizer_offload=False \
    actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16 \
    actor_rollout_ref.actor.deepspeed_config.mixed_precision=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.optim.lr=1e-5 \
    critic.ppo_mini_batch_size=256 \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.deepspeed_config.param_offload=False \
    critic.deepspeed_config.optimizer_offload=False \
    critic.deepspeed_config.model_dtype=bf16 \
    critic.deepspeed_config.mixed_precision=bf16 \
    trainer.total_epochs=2 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.experiment_name=deepspeed \
    trainer.project_name=fsdp_vs_deepspeed_benchmark \
    trainer.logger='["console"]' \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed \
    $@ 2>&1 | tee "$LOG_FILE"

echo "================================================"
echo "DeepSpeed PPO Benchmark Complete!"
echo "Results saved to: ./examples/benchmarks/fsdp_vs_deepspeed_ppo/results/deepspeed"
echo "Log saved to: $LOG_FILE"
echo "================================================"
