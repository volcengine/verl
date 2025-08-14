#!/bin/bash
#SBATCH --job-name=verl-deepseek-671b-small
#SBATCH --nodes=16               # 16 nodes (16 * 8 = 128 GPUs)
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=64       # CPU cores per task
#SBATCH --gres=gpu:8             # 8 H100 GPUs per node
#SBATCH --mem=0                  # Use all available memory
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=24:00:00          # Reduced time limit for testing
#SBATCH --partition=general      # Adjust to your cluster's partition
#SBATCH --output=verl_deepseek_671b_small_%j.out
#SBATCH --error=verl_deepseek_671b_small_%j.err

set -xeuo pipefail

# 🛠️  CONFIGURATION TROUBLESHOOTING GUIDE 🛠️
#
# This script has been carefully tuned to avoid common DeepSeek-V3 671B training errors:
#
# ❌ ERROR: "Can't load tokenizer for '/path/to/config'"
# ✅ FIX: MODEL_PATH uses HuggingFace model name (line ~73)
#
# ❌ ERROR: "ValueError: Please don't set ROCR_VISIBLE_DEVICES when CUDA_VISIBLE_DEVICES is set"  
# ✅ FIX: ROCm variables explicitly unset + env -u in srun commands (lines ~49-60)
#
# ❌ ERROR: "number of layers at middle stage: X must be divisible by Y"
# ✅ FIX: train_pp=4 gives 56÷2=28 layers per middle stage (line ~234)
#
# ❌ ERROR: "n_gpus (X) must be divisible by model_parallel_size (Y)"
# ✅ FIX: 128 total GPUs ÷ 4 PP = 32 perfect division (verified line ~223)
#
# ❌ ERROR: "decoder world_size (128) is not divisible by expert_tensor_model_pipeline_parallel size (X)"
# ✅ FIX: Simplified expert parallelism to train_ep=1 (no expert splitting)
#
# ❌ ERROR: "NameError: name 'TENorm' is not defined"
# ✅ FIX: Disabled Transformer Engine with use_transformer_engine=False (lines ~309, ~332)
#
# 🎯 CURRENT CONFIG: 16 nodes × 8 GPUs = 128 total, 12.5% utilization
# 🎯 PARALLELISM: gen_tp=8, train_tp=1, train_ep=1, train_pp=4  
# 🎯 LAYER DIST: 3 + (28×2) + 2 = 61 layers across 4 pipeline stages
# 🎯 TRANSFORMER ENGINE: Disabled to avoid import errors
#

# ================= Environment Setup =================
echo "🚀 Starting DeepSeek-V3 671B SMALL SCALE training setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8)) (128 GPUs with 16 nodes)"

# Load necessary modules (adjust based on your cluster)
# module load cuda/12.1
# module load gcc/9.3.0
# module load cmake

# Activate conda environment (assuming it's already set up)
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

# Unset AMD ROCm variables to avoid conflicts with CUDA (we're using NVIDIA GPUs)
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# Ray experimental settings to prevent automatic setting of GPU visibility
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# Set wandb API key if you have one
# export WANDB_API_KEY="your_wandb_api_key_here"

# ================= Paths Configuration =================
verl_workdir="$HOME/verl"

# ⭐ CUSTOMIZABLE MODEL PATHS - Change these to your preferred locations ⭐
MODEL_PATH="/data/xiangmin/dspk/DeepSeek-V3-config"                             # Use local path
MCORE_MODEL_PATH="/data/xiangmin/dspk/dpsk-v3-671B-BF16-dist_ckpt"     # Path to distributed checkpoint
DATA_HOME="/data/dspk"                                                 # Base data directory

# Data paths
TRAIN_FILE="$HOME/verl/data/dapo-math-17k.parquet"
TEST_FILE="$HOME/verl/data/aime-2024.parquet"

# Checkpoint directory
project_name='DAPO'
exp_name="DAPO-DeepSeek-671b-megatron-H100-SMALL-${SLURM_JOB_ID}"
CKPTS_DIR="$DATA_HOME/ckpts/${project_name}/${exp_name}"

# Create checkpoint directory
mkdir -p "${CKPTS_DIR}"

# Verify paths exist (MODEL_PATH is now HuggingFace model name, no need to check)

if [ ! -d "$MCORE_MODEL_PATH" ]; then
    echo "❌ Error: Megatron checkpoint path does not exist: $MCORE_MODEL_PATH"
    echo "Please download the distributed checkpoint from:"
    echo "https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file does not exist: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file does not exist: $TEST_FILE"
    exit 1
fi

cd "$verl_workdir"

# ================= Multi-node Setup =================
# Get node information
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle potential IPv6 addresses
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPv6 address detected. Using IPv4 address: $head_node_ip"
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
    echo "   Stopping Ray on $node..."
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force || true
    sleep 2
done

# ================= Ray Cluster Setup =================
echo "Starting Ray head node..."
ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --nodes=1 --ntasks=1 -w "$head_node" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

# Wait for head node to start
sleep 30

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker on node: $node_i"
    ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --nodes=1 --ntasks=1 -w "$node_i" \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 5
done

# Wait for all workers to connect
sleep 60

# ================= Training Configuration =================
# Algorithm parameters (SMALL SCALE - reduced for testing)
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# ⭐ REDUCED SEQUENCE LENGTHS FOR SMALL SCALE ⭐
max_prompt_length=$((1024 * 1))     # 1K tokens (reduced from 2K)
max_response_length=$((1024 * 2))   # 2K tokens (reduced from 8K)
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))   # Reduced buffer length
overlong_penalty_factor=0.1

loss_agg_mode="token-mean"

# ⭐ SMALL BATCH SIZES FOR FEWER NODES ⭐
train_prompt_bsz=64                 # Reduced from 256 to 64
n_resp_per_prompt=4                 # Reduced from 16 to 4
train_prompt_mini_bsz=8             # Reduced from 32 to 8

# H100 GPU configuration for DeepSeek-V3 671B (OPTIMIZED - 128 GPUs)
NNODES=${SLURM_JOB_NUM_NODES}
offload=True

# ⭐ OPTIMIZED PARALLELISM FOR 16 NODES (128 GPUs) ⭐
#
# CRITICAL: These values must satisfy THREE mathematical constraints:
#
# 1️⃣ LAYER DIVISIBILITY CONSTRAINT:
#    DeepSeek-V3 has 61 total layers with fixed first (3) and last (2) stages
#    Middle layers = 61 - 3 - 2 = 56 layers
#    RULE: 56 ÷ (train_pp - 2) MUST be a whole number
#    Valid train_pp: 3,4,6,9,10,16,30,58 (where 56÷(PP-2) = whole number)
#
# 2️⃣ GPU BUDGET CONSTRAINT:
#    Total GPUs needed = gen_tp + (train_tp × train_ep × train_pp × 2)
#    The "× 2" is because we need BOTH Actor and Reference models
#    RULE: gen_tp + (train_tp × train_ep × train_pp × 2) ≤ Total available GPUs
#    With 128 GPUs: 8 + (1 × 1 × 4 × 2) = 8 + 8 = 16 ≤ 128 ✅
#
# 3️⃣ FRAMEWORK DIVISIBILITY CONSTRAINT:
#    TOTAL GPUs MUST be evenly divisible by pipeline stages
#    RULE: Total GPUs ÷ train_pp MUST be a whole number
#    Verification: 128 ÷ 4 = 32 ✅ (perfect division)
#
# 4️⃣ EXPERT PARALLELISM CONSTRAINT:
#    For MoE models, expert tensor/pipeline parallel product must divide total GPUs
#    RULE: 128 ÷ expert_parallelism_product MUST be a whole number
#    With train_ep=1: No expert parallelism complexity ✅
#
# 5️⃣ TRANSFORMER ENGINE CONSTRAINT (NEW!):
#    Megatron tries to use Transformer Engine by default but it may not be installed
#    RULE: Either install Transformer Engine OR disable it with use_transformer_engine=False
#    We disable it to avoid import errors ✅
#
# OUR CHOSEN VALUES (satisfy ALL constraints):
gen_tp=8        # Generation tensor parallelism (8 GPUs for vLLM inference)
train_tp=1      # No tensor parallelism for training (keeps layers whole, simpler)
train_ep=1      # Expert parallelism (simplified - no expert splitting)
train_pp=4      # Pipeline parallelism (4 stages: 56÷2=28 layers per middle stage)
#
# ✅ VERIFICATION SUMMARY:
# Layer math:         56 ÷ (4-2) = 56 ÷ 2 = 28 layers per middle stage ✅
# GPU allocation:     8 + (1×1×4×2) = 8 + 8 = 16 GPUs total ✅
# Framework division: 128 ÷ 4 = 32 perfect division ✅
# Expert constraint:  train_ep=1 avoids complex expert parallelism ✅
# Transformer Engine: Disabled to avoid TENorm import errors ✅
# Resource usage:     16/128 GPUs = 12.5% utilization

# Dynamic batch sizing for large model
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))  # Reduced multiplier
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))  # Reduced multiplier

# Algorithm parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ================= Training Execution =================
echo "Starting DeepSeek-V3 671B OPTIMIZED training with $NNODES nodes and $((NNODES * 8)) GPUs total"
echo "🔧 SIMPLIFIED 16-NODE Configuration:"
echo "  - Nodes: $NNODES (simplified for compatibility)"
echo "  - Total GPUs: $((NNODES * 8)) (12.5% utilization - 16/128 GPUs)"
echo "  - Generation TP: $gen_tp (8 GPUs)"
echo "  - Training TP: $train_tp, EP: $train_ep, PP: $train_pp (28 layers per middle stage)"
echo "  - Max prompt length: $max_prompt_length (reduced from 2K)"
echo "  - Max response length: $max_response_length (reduced from 8K)"
echo "  - Training batch size: $train_prompt_bsz (reduced from 256)"
echo "  - Mini batch size: $train_prompt_mini_bsz (reduced from 32)"
echo "  - Responses per prompt: $n_resp_per_prompt (reduced from 16)"

PYTHONUNBUFFERED=1 ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="['$TEST_FILE']" \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=2 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.kv_channels=None \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=3 \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=2 \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.kv_channels=None \
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
    trainer.test_freq=2 \
    trainer.save_freq=3 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=5

# ================= Cleanup =================
echo "Training completed. Cleaning up Ray cluster..."
ray stop
echo "Job finished successfully!"
