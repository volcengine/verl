#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}


TRAIN_FILES=${TRAIN_FILES:-/data/xxx/train_multi.parquet}

backend=${BACKEND:-fsdp}

project_name=verl_sft_test

RESUME_MODE=auto

CKPT_HOME=${CKPT_HOME:-/data/xxx/saved_ckpts_naive}

MODEL_ID=${MODEL_ID:-/data/xxx/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-2}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}


PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_scheduler_type=cosine \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"

ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
echo "Using fsdp engine"
exp_name=gsm8k-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp${FSDP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}

mkdir -p "${CKPT_HOME}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=96 \
    data.max_length=32768 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=65536 \
    data.messages_key=messages \
    model.path=$MODEL_ID \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=4000 \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=5 \
    checkpoint.save_contents=[model,optimizer,extra]
