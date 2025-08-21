#!/bin/bash
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: sft_train_collabllm.sh <save_path> [other_configs...]"
    exit 1
fi

save_path=$1

# Shift the arguments so $@ refers to the rest
shift 1

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/collabllm-math-hard/train.parquet \
    data.val_files=$HOME/data/collabllm-math-hard/validation.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-2.5-0.5b-instruct-collabllm \
    trainer.logger=console \
    trainer.total_training_steps=1 $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true