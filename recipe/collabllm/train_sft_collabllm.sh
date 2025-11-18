#!/bin/bash
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: sft_train_collabllm.sh [<nproc_per_node> other_configs...]"
    exit 1
fi

nproc_per_node=$1

# Shift the arguments so $@ refers to the rest
shift 1

DATASET=math-hard-large

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/collabllm-$DATASET/sft_train.parquet \
    data.val_files=$HOME/data/collabllm-$DATASET/sft_validation.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    optim.lr=1e-6 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8196 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.project_name=collabllm-sft-$DATASET \
    trainer.experiment_name=collabllm-sft-qwen2.5-7B-$DATASET \
    trainer.logger=console \
    trainer.total_epochs=3 $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true $@