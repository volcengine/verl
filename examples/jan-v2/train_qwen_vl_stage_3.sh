#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.
set -o pipefail # Return the exit status of the last command in the pipe that failed.

# ========================================================================================
# R-HORIZON: Distributed Reinforcement Learning Training Script
#
# This script launches a distributed training job using Ray and the GRPO algorithm.
# It is designed to be run in a multi-GPU, multi-node environment.
#
# Usage:
#   - For single-node training: bash scripts/train_rl.sh
#   - For multi-node training: Ensure the Ray cluster is initialized, then run this
#     script on each node.
#   - Customize parameters by passing them as arguments, e.g.,
#     bash scripts/train_rl.sh --model_path "/path/to/your/model" --output_dir "/path/to/save"
# ========================================================================================

# ---
# ⚙️ Default Configuration
# ---
# These can be overridden by command-line arguments.
MODEL_PATH="/mnt/nas/alex/verl/converted-model/r-horizon-qwen3-vl-8b-vision-mix-rhorizon-k-234/checkpoint-20" # Path to your base model
TRAIN_DATA_DIR="/mnt/nas/data/r-horizon-training-data" # Directory for training data
EVAL_DATA_DIR="/mnt/nas/data/r-horizon-training-data"
OUTPUT_DIR="./checkpoints/r-horizon-rl-qwen3-vl-8b-vision-mix-rhorizon-k-234-stage-3" # Directory to save checkpoints and logs

WORLD_SIZE=1  # Number of nodes
GPUS_PER_NODE=8 # Number of GPUs per node
MASTER_PORT=29500

# ---
# ↗️ Command-line Argument Parsing
# ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --train_data_dir) TRAIN_DATA_DIR="$2"; shift ;;
        --eval_data_dir) EVAL_DATA_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --world_size) WORLD_SIZE="$2"; shift ;;
        --gpus_per_node) GPUS_PER_NODE="$2"; shift ;;
        --master_port) MASTER_PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ---
# 🛠️ Environment Setup
# ---
# Set PyTorch and NCCL environment variables for performance and debugging.
export WANDB_API_KEY=
export TORCH_CPP_LOG_LEVEL="INFO"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export NCCL_DEBUG="WARN"
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
# export VLLM_ATTENTION_BACKEND="XFORMERS"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export NCCL_P2P_DISABLE=1
# export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_USE_V1=1


# ---
# 🚀 Training Hyperparameters
# ---
# Rollout and PPO settings
ROLLOUT_BATCH_SIZE=48
PPO_MINI_BATCH=48
micro_batch_size_per_gpu=8
micro_batch_logprob_size_per_gpu=16
MAX_PROMPT_LENGTH=2048
RESPONSE_LENGTH=24000 # Renamed from RES_LENGTH for clarity
GROUP_SIZE=8
N_VAL_SAMPLES=8
TRAIN_TEMPERATURE=1.
WANDB_MODE='online'

# Tensor/Sequence Parallelism (for very large models)
TP=1 # Tensor Parallelism
SP=4 # Sequence Parallelism
MAX_TOKEN_LEN=$(((RESPONSE_LENGTH + MAX_PROMPT_LENGTH + 1000) / SP))

# ---
# 📊 Dataset Configuration
# ---
# Assumes data files are in the specified directories.
# Modify the file names if your dataset structure is different.
train_files="[\"/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-text-only-single-train/train_k2.parquet\",\"/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-text-only-single-train/train_k3.parquet\",\"/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-text-only-single-train/train_k4.parquet\",\"/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-v2/train.parquet\",\"/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-v2/train_k3.parquet\",\"/mnt/nas/bachvd/Code-Agent/verl/data/r-horizon-vision-v2/train_k4.parquet\"]"
test_files="[\"$EVAL_DATA_DIR/aime24.parquet\",\"$EVAL_DATA_DIR/aime25.parquet\"]"

# ---
#  wandb Configuration (optional)
# ---
# Set to "online" to enable Weights & Biases logging.
# Ensure you have run `wandb login` first.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"

# ---

# Create directories
mkdir -p "$OUTPUT_DIR"
STATS_DIR="${OUTPUT_DIR}/stats"
mkdir -p "$STATS_DIR"

# Project and Experiment Naming
PROJECT_NAME="Qwen3-VL-8B-R-HORIZON-RL-Training-k-234"
EXP_NAME="grpo-vision-mix-text-stage-3-$(basename ${MODEL_PATH})-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Submitting Ray job..."
echo "  - Model: ${MODEL_PATH}"
echo "  - Output Dir: ${OUTPUT_DIR}"
echo "  - Train Data: ${train_files}"
echo "  - Eval Data: ${test_files}"

# Submit the training job to the Ray cluster.
# The entry point is assumed to be `verl.trainer.main_ppo`.
# The configuration is passed using Hydra-style overrides.
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.use_reward_clip=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.nccl_timeout=3600 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=0.1 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-sum-norm' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=26048 \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.top_p=0.95 \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    +actor_rollout_ref.rollout.presence_penalty=1.5 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_logprob_size_per_gpu \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.test_freq=1000 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=5 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=10

