#!/bin/bash
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: sft_train_collabllm.sh [<nproc_per_node> other_configs...]"
    exit 1
fi

nproc_per_node=$1

# Shift the arguments so $@ refers to the rest
shift 1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/collabllm-math-hard/sft_train.parquet \
    data.val_files=$HOME/data/collabllm-math-hard/sft_validation.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    optim.lr=1e-6 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.project_name=collabllm-sft \
    trainer.experiment_name=multiturn-sft-qwen-2.5-7b-instruct-collabllm \
    trainer.logger=console \
    trainer.total_epochs=3 $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true $@