#!/bin/bash
#SBATCH --job-name=verl-qwen25-math-7b
#SBATCH --nodes=4                # 4 nodes (4 * 8 = 32 GPUs) - matches original NNODES=4
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=64       # CPU cores per task
#SBATCH --gres=gpu:8             # 8 GPUs per node (H100/A100)
#SBATCH --mem=0                  # Use all available memory
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=12:00:00          # 12 hours should be sufficient for 7B model
#SBATCH --partition=general      # Adjust to your cluster's partition
#SBATCH --output=verl_qwen25_math_7b_%j.out
#SBATCH --error=verl_qwen25_math_7b_%j.err

set -xeuo pipefail

# 🛠️  QWEN2.5-MATH-7B SLURM CONFIGURATION 🛠️
#
# This script is optimized for Qwen2.5-Math-7B model training:
#
# 🎯 MODEL: Qwen2.5-Math-7B (7 billion parameters)
# 🎯 HARDWARE: 4 nodes × 8 GPUs = 32 total GPUs
# 🎯 PARALLELISM: gen_tp=4, train_tp=4, train_pp=2 (matches original config)
# 🎯 RESOURCES: Much more efficient than 671B - higher GPU utilization
#
# Key differences from DeepSeek-671B script:
# - Smaller model = fewer nodes needed (4 vs 16)
# - Higher GPU utilization (62.5% vs 12.5%)
# - Simpler parallelism strategy
# - Faster training times expected

# ================= Environment Setup =================
echo "🚀 Starting Qwen2.5-Math-7B DAPO training setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8)) (32 GPUs with 4 nodes)"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate verl

# Environment variables for optimization
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Unset AMD ROCm variables to avoid conflicts
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# Ray experimental settings
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# ================= Paths Configuration =================
verl_workdir="$HOME/verl"

# Model and data paths (adjust to your setup)
MODEL_PATH="${HOME}/verl/models/Qwen2.5-Math-7B"
DATA_HOME="${HOME}/verl"
TRAIN_FILE="${DATA_HOME}/data/dapo-math-17k.parquet"
TEST_FILE="${DATA_HOME}/data/aime-2024.parquet"

# Checkpoint directory
project_name='DAPO'
exp_name="DAPO-Qwen2.5-7b-MATH-megatron-${SLURM_JOB_ID}"
CKPTS_DIR="${DATA_HOME}/ckpts/${project_name}/${exp_name}"

mkdir -p "${CKPTS_DIR}"
cd "$verl_workdir"

# ================= Multi-node Setup =================
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle IPv6 addresses
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Ray head: $ip_head"

# ================= Cleanup Existing Ray Sessions =================
echo "🧹 Cleaning up any existing Ray sessions..."
for node in "${nodes_array[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force || true
    sleep 2
done

# ================= Ray Cluster Setup =================
echo "Starting Ray head node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

sleep 30

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker on node: $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 5
done

sleep 60

# ================= Training Configuration =================
# EXACT SAME AS ORIGINAL 7B SCRIPT
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Original 7B sequence lengths (FULL SCALE)
max_prompt_length=$((1024 * 2))     # 2K tokens
max_response_length=$((1024 * 8))   # 8K tokens
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Original 7B batch sizes (FULL SCALE)
train_prompt_bsz=512
n_resp_per_prompt=16
train_prompt_mini_bsz=32

# GPU configuration
NNODES=${SLURM_JOB_NUM_NODES}
offload=True

# PARALLELISM (EXACT MATCH TO ORIGINAL 7B)
# GPU Usage: 4 + (4 × 2 × 2) = 20 GPUs out of 32 available (62.5% utilization)
gen_tp=4        # Generation tensor parallelism
train_tp=4      # Training tensor parallelism  
train_pp=2      # Pipeline parallelism

# Dynamic batch sizing
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

# Algorithm parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ================= Training Execution =================
echo "Starting Qwen2.5-Math-7B DAPO training with $NNODES nodes and $((NNODES * 8)) GPUs total"
echo "🔧 7B Model Configuration (matching original script):"
echo "  - Nodes: $NNODES"
echo "  - Total GPUs: $((NNODES * 8)) (62.5% utilization - 20/32 GPUs)"
echo "  - Generation TP: $gen_tp (4 GPUs)"
echo "  - Training TP: $train_tp, PP: $train_pp (16 GPUs for actor+ref)"

PYTHONUNBUFFERED=1 env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10

# ================= Cleanup =================
echo "Training completed. Cleaning up Ray cluster..."
ray stop
echo "Job finished successfully!"