#!/bin/bash
#SBATCH --partition=plm
#SBATCH --job-name=xuekai_gspo_7b
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/flowrl_%j.out
#SBATCH --error=logs/flowrl_%j.err

# Unset AMD GPU variable to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

export RAY_TMPDIR=/mnt/hwfile/linzhouhan/rt
mkdir -p "$RAY_TMPDIR"

proxy_on

# Print GPU info
nvidia-smi

# Run the training script
bash recipe/flowrl/run_gspo_qwen2.5_7b.sh