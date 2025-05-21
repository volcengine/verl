#!/bin/bash
# SFT script for Qwen2.5-VL-7B on Geo3K dataset

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_vl_7b_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Export WANDB API key if needed
# export WANDB_API_KEY=your_api_key_here

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    data.prompt_key=prompt \
    data.response_key=reward_model \
    data.response_dict_keys=['ground_truth'] \
    data.image_key=images \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    optim.lr=1e-6 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=geo3k-sft \
    trainer.experiment_name=geo3k-sft-qwen2.5-vl-7b-instruct \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.default_hdfs_dir=null \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear $@