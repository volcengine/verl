#!/bin/bash
#SBATCH --job-name=qwen2_5_1_5b_grpo
#SBATCH --partition=main              
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8                 
#SBATCH --cpus-per-task=64          
#SBATCH --mem=500G                   
#SBATCH --time=72:00:00              
#SBATCH --output=/mnt/weka/home/haolong.jia/RL/logs/qwen2_5_grpo.out
#SBATCH --error=/mnt/weka/home/haolong.jia/RL/logs/qwen2_5_grpo.err


source /mnt/weka/home/haolong.jia/miniconda3/bin/activate verl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=$((RANDOM % 10000 + 20000))
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost

# WandB configuration
export WANDB_API_KEY="7a43277c376f2b14ab11f153f74e8448b07aac7c"
export WANDB_PROJECT="RL"  
export WANDB_ENTITY="haolong"  

cd /mnt/weka/home/haolong.jia/RL/training/verl

bash custom_rl/scripts/run_qwen2_5_1_5b_gsm8k.sh