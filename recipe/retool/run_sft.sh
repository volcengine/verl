#!/bin/bash
set -x

DATA_HOME=${DATA_HOME:-"${HOME}/verl"}

mkdir $DATA_HOME/data

wget -O $DATA_HOME/data/retool_sft_dataset.parquet https://huggingface.co/datasets/vermouth1992/ReTool-SFT/blob/main/data/train-00000-of-00001.parquet

TRAIN_DATA=$DATA_HOME/data/retool_sft_dataset.parquet
EVAL_DATA=$DATA_HOME/data/retool_sft_dataset.parquet
MODEL_PATH=$DATA_HOME/models/Qwen3-8B
SAVE_PATH=./

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-3-8b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true