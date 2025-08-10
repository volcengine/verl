#!/bin/bash
#SBATCH --job-name=verl-dapo-multinode
#SBATCH --nodes=4                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=64       # CPU cores per task
#SBATCH --gres=gpu:8             # 8 GPUs per node
#SBATCH --mem=0                  # Use all available memory
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=24:00:00          # 24 hour time limit
#SBATCH --partition=general
#SBATCH --output=verl_dapo_multinode_%j.out
#SBATCH --error=verl_dapo_multinode_%j.err

set -xeuo pipefail

# ================= Environment Setup =================
echo "🚀 Starting multi-node DAPO training setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"

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
RAY_DATA_HOME="$HOME/verl"
MODEL_PATH="$RAY_DATA_HOME/models/Qwen3-30B-A3B-Base"
TRAIN_FILE="$RAY_DATA_HOME/data/dapo-math-17k.parquet"
TEST_FILE="$RAY_DATA_HOME/data/aime-2024.parquet"

# Verify paths exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training file does not exist: $TRAIN_FILE"
    exit 1
fi

cd "$verl_workdir"

# ================= Ray Cluster Setup =================
# Get node information
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
    echo "IPv6 address detected. Using IPv4 address: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "🌐 Ray cluster setup:"
echo "   Head node: $head_node"
echo "   Head node IP: $head_node_ip"
echo "   Ray address: $ip_head"
echo "   Total nodes: ${#nodes_array[@]}"

# ================= Cleanup Existing Ray Sessions =================
echo "🧹 Cleaning up any existing Ray sessions..."
for node in "${nodes_array[@]}"; do
    echo "   Stopping Ray on $node..."
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force || true
    sleep 2
done

# ================= Start Ray Head Node =================
echo "🚀 Starting Ray head node on $head_node..."
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
    --no-monitor

# Wait for head node to be ready
sleep 15

# ================= Start Ray Worker Nodes =================
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "🔗 Starting $worker_num worker nodes..."

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "   Starting worker $i on $node_i..."
    ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --nodes=1 --ntasks=1 -w "$node_i" \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "8" \
        --no-monitor
    sleep 5
done

# Wait for all nodes to join
echo "⏳ Waiting for all nodes to join the cluster..."
sleep 30

# Verify cluster status
echo "📊 Ray cluster status:"
srun --nodes=1 --ntasks=1 -w "$head_node" ray status

# ================= Training Configuration =================
project_name='DAPO-SLURM-MultiNode'
exp_name="DAPO-Qwen3-30B-${SLURM_JOB_NUM_NODES}Nodes-MATH-${SLURM_JOB_ID}"
CKPTS_DIR="$RAY_DATA_HOME/ckpts/${project_name}/${exp_name}"

# Create checkpoint directory
mkdir -p "$CKPTS_DIR"

# ================= Start Training =================
echo "🎯 Starting DAPO training..."
echo "   Project: $project_name"
echo "   Experiment: $exp_name"
echo "   Checkpoints: $CKPTS_DIR"

PYTHONUNBUFFERED=1 ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=1536 \
    data.max_response_length=3072 \
    data.train_batch_size=64 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=32 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.1 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=4096 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$exp_name" \
    trainer.n_gpus_per_node="8" \
    trainer.nnodes="$SLURM_JOB_NUM_NODES" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=300 \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 2>&1 | tee verl_dapo_multinode.log

echo "🎉 Multi-node DAPO training completed!"
echo "📊 Dashboard was available at: http://$head_node_ip:8265"
echo "📁 Checkpoints saved to: $CKPTS_DIR"

