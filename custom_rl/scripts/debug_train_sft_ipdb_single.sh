#!/bin/bash
set -x

# Set environment variables for single process distributed training
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Enable debugging features
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

save_path=/mnt/sharefs/users/haolong.jia/RL-model/sft_rl_debug

# Run with ipdb for better debugging experience
python -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/sharefs/users/haolong.jia/RL-data/char_count/sft/train.parquet \
    data.val_files=/mnt/sharefs/users/haolong.jia/RL-data/char_count/sft/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=256 \
    data.train_batch_size=2 \
    use_remove_padding=True \
    model.partial_pretrain=HuggingFaceTB/SmolLM2-135M-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=char_count-sft-debug \
    trainer.experiment_name=debug-run \
    trainer.total_epochs=1 \
    trainer.test_freq=10 \
    trainer.save_freq=100 \
    trainer.logger=console 