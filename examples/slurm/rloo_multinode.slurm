#!/bin/bash
#SBATCH --job-name=verl-rloo-multinode
#SBATCH --nodes=4                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=64       # CPU cores per task
#SBATCH --gres=gpu:8             # 8 GPUs per node
#SBATCH --mem=0                  # Use all available memory
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=12:00:00          # 12 hour time limit
#SBATCH --partition=general
#SBATCH --output=verl_rloo_multinode_%j.out
#SBATCH --error=verl_rloo_multinode_%j.err

set -xeuo pipefail

# ================= Configuration =================
MODEL_PATH="$HOME/verl/models/Qwen3-30B-A3B-Base"
TRAIN_FILE="$HOME/verl/data/gsm8k/train.parquet"
TEST_FILE="$HOME/verl/data/gsm8k/test.parquet"

PROJECT_NAME="VERL-RLOO-MultiNode"
EXP_NAME="RLOO-Qwen3-30B-A3B-Base-${SLURM_JOB_NUM_NODES}Nodes-${SLURM_JOB_ID}"

# ================= Environment Setup =================
echo "🚀 Starting multi-node RLOO training setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate verl

# Environment variables
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Ray settings
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# Unset ROCm variables
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true
export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# ================= Path Setup =================
VERL_WORKDIR="$HOME/verl"
CKPTS_DIR="$HOME/verl/ckpts/${PROJECT_NAME}/${EXP_NAME}"

# Verify paths
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file does not exist: $TRAIN_FILE"
    exit 1
fi

mkdir -p "$CKPTS_DIR"
cd "$VERL_WORKDIR"

# ================= Ray Cluster Setup =================
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Handle IPv6
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPv6 detected. Using IPv4: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "🌐 Ray cluster:"
echo "   Head: $head_node ($head_node_ip)"
echo "   Address: $ip_head"
echo "   Nodes: ${#nodes_array[@]}"

# Stop existing Ray sessions
echo "🧹 Cleaning up Ray sessions..."
for node in "${nodes_array[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force || true
done
sleep 5

# Start head node
echo "🚀 Starting Ray head node..."
ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --nodes=1 --ntasks=1 -w "$head_node" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ray start --head \
    --node-ip-address="$head_node_ip" \
    --port=$port \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus "8" \
    --disable-usage-stats \
    --no-monitor &

sleep 15

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "🔗 Starting $worker_num workers..."

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "   Worker $i: $node_i"
    ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --nodes=1 --ntasks=1 -w "$node_i" \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "8" \
        --no-monitor &
    sleep 3
done

sleep 20

# Verify cluster
echo "📊 Cluster status:"
srun --nodes=1 --ntasks=1 -w "$head_node" ray status

# ================= Start RLOO Training =================
echo "🎯 Starting RLOO training..."
echo "   Project: $PROJECT_NAME"
echo "   Experiment: $EXP_NAME"
echo "   Checkpoints: $CKPTS_DIR"

PYTHONUNBUFFERED=1 ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="$SLURM_JOB_NUM_NODES" \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    2>&1 | tee "verl_rloo_multinode_${SLURM_JOB_ID}.log"

# ================= Cleanup =================
echo "🎉 RLOO training completed!"
echo "📊 Dashboard: http://$head_node_ip:8265"
echo "📁 Checkpoints: $CKPTS_DIR"
echo "📋 Log: verl_rloo_multinode_${SLURM_JOB_ID}.log"

# Stop Ray
echo "🧹 Stopping Ray cluster..."
for node in "${nodes_array[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force || true
done

echo "✅ Job completed!"
