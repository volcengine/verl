#!/bin/bash
#SBATCH --job-name=verl-qwen3-236b
#SBATCH --nodes=16               # Number of nodes (16 nodes * 8 H100 = 128 GPUs)
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=64       # CPU cores per task
#SBATCH --gres=gpu:8             # 8 H100 GPUs per node
#SBATCH --mem=0                  # Use all available memory
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=48:00:00          # 48 hour time limit
#SBATCH --partition=general      # Adjust to your cluster's partition
#SBATCH --output=verl_qwen3_236b_%j.out
#SBATCH --error=verl_qwen3_236b_%j.err

set -xeuo pipefail

# ================= Environment Setup =================
echo "🚀 Starting Qwen3-236B training setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"

# Load necessary modules (adjust based on your cluster)
# module load cuda/12.1
# module load gcc/9.3.0
# module load cmake

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

# Ray experimental settings
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# Unset AMD ROCm variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true
export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# Set wandb API key if you have one
# export WANDB_API_KEY="your_wandb_api_key_here"

# ================= Paths Configuration =================
verl_workdir="$HOME/verl"
RAY_DATA_HOME="$HOME/verl"

# Model paths
MODEL_PATH="$RAY_DATA_HOME/models/Qwen3-235B-A22B"
MCORE_MODEL_PATH="$RAY_DATA_HOME/models/Qwen3-235B-A22B_dist_ckpt_mcore/"

# Data paths
TRAIN_FILE="$RAY_DATA_HOME/dataset/dapo-math-17k.parquet"
TEST_FILE="$RAY_DATA_HOME/dataset/aime-2024.parquet"

# Checkpoint directory
project_name='DAPO'
exp_name="DAPO-Qwen3-236b-megatron-H100-${SLURM_JOB_ID}"
CKPTS_DIR="$RAY_DATA_HOME/ckpt/${project_name}/${exp_name}"

# Verify paths exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    echo "Please download the model first using the setup script"
    exit 1
fi

if [ ! -d "$MCORE_MODEL_PATH" ]; then
    echo "❌ Error: Megatron checkpoint path does not exist: $MCORE_MODEL_PATH"
    echo "Please convert the model first using the conversion script"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file does not exist: $TRAIN_FILE"
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

# ================= Ray Cluster Setup =================
echo "Starting Ray head node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

# Wait for head node to start
sleep 30

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker on node: $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 5
done

# Wait for all workers to connect
sleep 60

# ================= Training Configuration =================
# Algorithm parameters
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=0.1

loss_agg_mode="token-mean"

train_prompt_bsz=256
n_resp_per_prompt=4
train_prompt_mini_bsz=16

# H100 GPU configuration (optimized for 80GB memory)
NNODES=${SLURM_JOB_NUM_NODES}
offload=True
gen_tp=8        # Generation tensor parallelism
train_tp=4      # Training tensor parallelism  
train_ep=4      # Expert parallelism
train_pp=8      # Pipeline parallelism

# Algorithm parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ================= Training Execution =================
echo "Starting Qwen3-236B training with $NNODES nodes and $((NNODES * 8)) GPUs total"

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=5 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=5 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
    trainer.save_freq=20 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10

# ================= Cleanup =================
echo "Training completed. Cleaning up Ray cluster..."
ray stop
echo "Job finished successfully!"
