#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

NUM_GPUS=${NUM_GPUS:-8}

TRAIN_FILES=~/data/gsm8k_sft/train.parquet
VAL_FILES=~/data/gsm8k_sft/test.parquet

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.pad_mode=left_right \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=8192 \
    model.path=/mnt/hdfs/zhangchi.usc1992_lf_lq/models/Qwen2.5-3B-Instruct \
    engine=fsdp \
    engine.optim.lr=1e-5 \
    engine.optim.lr_warmup_steps_ratio=0.05 \
    engine.optim.weight_decay=0.1 \
    engine.optim.betas="[0.9,0.95]" \
    engine.optim.clip_grad=1.0 \
    engine.optim.min_lr_ratio=0.1 \
    engine.optim.warmup_style=cosine \
    engine.engine.ulysses_sequence_parallel_size=2 \
    engine.engine.strategy=fsdp


    # data.multiturn.messages_key=messages \
    # data.micro_batch_size_per_gpu=${micro_bsz} \
    # trainer.default_local_dir="${ckpts_home}" \
    # trainer.project_name="${project_name}" \
    # trainer.experiment_name="${exp_name}" \
    # trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    # trainer.save_freq=${SAVE_FREQ} \
    # trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    # trainer.max_ckpt_to_keep=1 \
    # trainer.resume_mode=${RESUME_MODE} \
    # trainer.logger=['console'] $@