#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}

TRAIN_FILES=gsm8k/train.parquet
VAL_FILES=gsm8k/test.parquet

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    model.path=/mnt/hdfs/zhangchi.usc1992_lf_lq/models/Qwen2.5-0.5B-Instruct \
    engine=fsdp

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