#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script to run PPO training with LLM-as-Judge reward scoring.

This script demonstrates how to configure verl to use LLM-as-Judge
for reward computation during PPO training.

Prerequisites:
1. Set your API key as environment variable:
   export OPENAI_API_KEY="sk-..."

2. Create a dataset with questions and reference answers:
   - The dataset should have 'prompt' and 'answer' fields
   - See examples/data for dataset format examples

Usage:
    # Basic usage with default settings
    bash run_llm_judge_ppo.sh

    # Custom model and configuration
    bash run_llm_judge_ppo.sh --model-name /path/to/model --num-gpus 4

    # Use Azure OpenAI
    AZURE_OPENAI_KEY="..." bash run_llm_judge_ppo.sh --api-url "https://your-resource.openai.azure.com/..."

    # Use different judge model
    bash run_llm_judge_ppo.sh --judge-model gpt-3.5-turbo
"""

set -euo pipefail

# ============================================
# Configuration
# ============================================

# Model settings
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"  # Default model
NUM_GPUS="${NUM_GPUS:-1}"  # Number of GPUs to use
MAX_ROLLOUT_LEN="${MAX_ROLLOUT_LEN:-512}"  # Max rollout length

# LLM-as-Judge settings
API_KEY="${OPENAI_API_KEY:-}"  # API key (set OPENAI_API_KEY env var)
API_URL="${API_URL:-https://api.openai.com/v1/chat/completions}"  # Default: OpenAI
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4}"  # Model to use for judgment
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"  # Max concurrent API requests
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.1}"  # Lower for more consistent scoring

# Training settings
TRAINING_STEPS="${TRAINING_STEPS:-100}"  # Number of training steps
BATCH_SIZE="${BATCH_SIZE:-16}"  # Batch size
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"  # Gradient accumulation steps
LEARNING_RATE="${LEARNING_RATE:-1e-6}"  # Learning rate

# Dataset settings
DATASET_PATH="${DATASET_PATH:-examples/data/custom_dataset.jsonl}"  # Path to dataset
NUM_EXAMINE="${NUM_EXAMINE:-1}"  # Number of samples to examine per step

# Output settings
OUTPUT_DIR="${OUTPUT_DIR:-outputs/llm_judge_ppo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-llm_judge_experiment}"

# ============================================
# Validation
# ============================================

if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set!"
    echo ""
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    echo "Or use Azure:"
    echo "  AZURE_OPENAI_KEY='...' bash $0 --api-url 'https://...'"
    exit 1
fi

echo "============================================"
echo "LLM-as-Judge PPO Training Configuration"
echo "============================================"
echo "Model:            $MODEL_NAME"
echo "GPUs:             $NUM_GPUS"
echo "Max Rollout:      $MAX_ROLLOUT_LEN"
echo ""
echo "Judge Model:      $JUDGE_MODEL"
echo "Judge API URL:    $API_URL"
echo "Max Concurrent:    $MAX_CONCURRENT"
echo ""
echo "Training Steps:    $TRAINING_STEPS"
echo "Batch Size:       $BATCH_SIZE"
echo "Learning Rate:     $LEARNING_RATE"
echo ""
echo "Dataset:          $DATASET_PATH"
echo "Output:           $OUTPUT_DIR"
echo "============================================"

# ============================================
# Run PPO Training with LLM-as-Judge
# ============================================

python -m verl.trainer.main_ppo \
    trainer.ppo_train.data.train_files=$DATASET_PATH \
    trainer.ppo_train.data.val_files=$DATASET_PATH \
    trainer.ppo_train.data.train_batch_size=$BATCH_SIZE \
    trainer.ppo_train.data.max_prompt_length=1024 \
    trainer.ppo_train.data.max_response_length=$MAX_ROLLOUT_LEN \
    \
    trainer.ppo_train.rollout.name=vllm \
    trainer.ppo_train.rollout.tensor_model_parallel_size=$NUM_GPUS \
    trainer.ppo_train.rollout.model.path=$MODEL_NAME \
    \
    trainer.ppo_train.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    trainer.ppo_train.actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    \
    trainer.ppo_train.actor_rollout_ref.actor.strategy=fsdp \
    trainer.ppo_train.actor_rollout_ref.actor.fsdp_config.param_offload=True \
    \
    trainer.ppo_train.reward_manager.source=custom \
    \
    trainer.ppo_train.reward_manager.custom_reward_function.path=$(pwd)/../../verl/utils/reward_score/llm_judge.py \
    trainer.ppo_train.reward_manager.custom_reward_function.name=compute_score_async \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.api_key=$API_KEY \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.base_url=$API_URL \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.model=$JUDGE_MODEL \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.max_tokens=100 \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.temperature=$JUDGE_TEMPERATURE \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.max_concurrent=$MAX_CONCURRENT \
    trainer.ppo_train.reward_manager.custom_reward_function.reward_kwargs.timeout=60 \
    \
    trainer.ppo_train.total_steps=$TRAINING_STEPS \
    trainer.ppo_train.project_dir=$OUTPUT_DIR \
    trainer.ppo_train.experiment_name=$EXPERIMENT_NAME \
    \
    trainer.logger=console \
    trainer.default_local_dir=$OUTPUT_DIR

echo ""
echo "============================================"
echo "Training completed!"
echo "Check results at: $OUTPUT_DIR"
echo "============================================"
