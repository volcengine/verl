#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

NUM_GPUS=${NUM_GPUS:-8}

TRAIN_FILES=~/data/gsm8k_sft/train.parquet
VAL_FILES=~/data/gsm8k_sft/test.parquet

backend=${BACKEND:-fsdp}

project_name=verl_sft_test
exp_name=gsm8k-${backend}
RESUME_MODE=disable

ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-${backend}}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=2 \
    engine.strategy=fsdp"


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=2 \
    engine.pipeline_model_parallel_size=2 \
    engine.virtual_pipeline_model_parallel_size=2 \
    engine.context_parallel_size=1"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
fi

mkdir -p "${ckpts_home}"

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
    data.messages_key=messages \
    model.path=$MODEL_PATH \
    ${ENGINE_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps=2 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \

    # trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    # trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    # trainer.max_ckpt_to_keep=1 \
    
rm -rf "${ckpts_home:?}/*"