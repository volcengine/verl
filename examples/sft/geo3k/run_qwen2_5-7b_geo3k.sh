#!/bin/bash
# SFT script for Qwen2.5-VL-7B on Geo3K dataset

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5-7b_geo3k.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nproc_per_node=$nproc_per_node --nnodes=1 \
  -m verl.trainer.fsdp_sft_trainer \
    algorithm.adv_estimator=sft \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    model.name_or_path=Qwen/Qwen2.5-VL-7B-Instruct \
    model.torch_dtype=float16 \
    model.device_map=auto \
    model.attn_implementation=sdpa \
    model.trust_remote_code=True \
    optimizer.lr=1e-6 \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.param_offload=False \
    model.fsdp_config.optimizer_offload=False \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.lora_target_modules=all-linear \
    trainer.default_local_dir=$save_path \
    trainer.project_name=geo3k-sft \
    trainer.experiment_name=geo3k-sft-qwen2.5-vl-7b-instruct \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.nnodes=1 \
    +trainer.save_freq=1 \
    +trainer.test_freq=1 \
    trainer.total_epochs=3 $@