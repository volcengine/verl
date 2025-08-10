#!/bin/bash
#SBATCH --job-name=verl-sft-multinode
#SBATCH --nodes=2                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=64       # CPU cores per task
#SBATCH --gres=gpu:8             # 8 GPUs per node
#SBATCH --mem=0                  # Use all available memory
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=24:00:00          # 24 hour time limit
#SBATCH --partition=general
#SBATCH --output=verl_sft_multinode_%j.out
#SBATCH --error=verl_sft_multinode_%j.err

set -xeuo pipefail

# ================= Environment Setup =================
echo "🚀 Starting multi-node SFT training setup..."
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate verl

# Environment variables for optimization
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Unset AMD ROCm variables to avoid conflicts with CUDA (we're using NVIDIA GPUs)
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# Set wandb API key if you have one
# export WANDB_API_KEY="your_wandb_api_key_here"

# ================= Paths Configuration =================
verl_workdir="$HOME/verl"
RAY_DATA_HOME="$HOME/verl"
MODEL_PATH="$RAY_DATA_HOME/models/Qwen3-30B-A3B-Base"
TRAIN_FILE="$RAY_DATA_HOME/data/aime-2024.parquet"
VAL_FILE="$RAY_DATA_HOME/data/aime-2024.parquet"

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

# ================= Multi-node Setup =================
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

# Set torchrun parameters
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * 8))  # 8 GPUs per node
export NODE_RANK=$SLURM_PROCID

echo "🌐 Multi-node setup:"
echo "   Head node: $head_node"
echo "   Head node IP: $head_node_ip"
echo "   Master address: $MASTER_ADDR:$MASTER_PORT"
echo "   Total nodes: ${#nodes_array[@]}"
echo "   World size: $WORLD_SIZE"

# ================= Training Configuration =================
project_name='SFT-SLURM-MultiNode'
exp_name="SFT-Qwen3-30B-${SLURM_JOB_NUM_NODES}Nodes-AIME-${SLURM_JOB_ID}"
CKPTS_DIR="$RAY_DATA_HOME/ckpts/${project_name}/${exp_name}"

# Create checkpoint directory
mkdir -p "$CKPTS_DIR"

# ================= Start SFT Training =================
echo "🎯 Starting SFT training..."
echo "   Project: $project_name"
echo "   Experiment: $exp_name"
echo "   Checkpoints: $CKPTS_DIR"

# Launch training on all nodes
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    bash -c "
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$WORLD_SIZE
    export NODE_RANK=\$SLURM_PROCID
    export ROCR_VISIBLE_DEVICES=\"\"
    export HIP_VISIBLE_DEVICES=\"\"
    
    cd $verl_workdir
    
    torchrun --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=8 \
        --master_addr=\$MASTER_ADDR \
        --master_port=\$MASTER_PORT \
        --node_rank=\$NODE_RANK \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=\"$TRAIN_FILE\" \
        data.val_files=\"$VAL_FILE\" \
        data.prompt_key=prompt \
        data.response_key=reward_model \
        data.prompt_dict_keys=['content'] \
        data.response_dict_keys=['ground_truth'] \
        data.max_length=4096 \
        data.train_batch_size=64 \
        data.micro_batch_size_per_gpu=2 \
        model.partial_pretrain=\"$MODEL_PATH\" \
        model.strategy=fsdp \
        model.enable_gradient_checkpointing=True \
        optim.lr=1e-5 \
        optim.warmup_steps_ratio=0.1 \
        optim.weight_decay=0.1 \
        optim.clip_grad=1.0 \
        optim.lr_scheduler=cosine \
        ulysses_sequence_parallel_size=4 \
        use_remove_padding=True \
        trainer.default_local_dir=\"$CKPTS_DIR\" \
        trainer.project_name=\"$project_name\" \
        trainer.experiment_name=\"$exp_name\" \
        trainer.logger='[\"console\",\"wandb\"]' \
        trainer.total_epochs=3 \
        trainer.save_freq=100 \
        trainer.max_ckpt_to_keep=3 \
        trainer.resume_mode=auto \
        trainer.test_freq=50
    " 2>&1 | tee verl_sft_multinode.log

echo "🎉 Multi-node SFT training completed!"
echo "📁 Checkpoints saved to: $CKPTS_DIR"
