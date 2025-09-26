#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

NUM_GPUS=${NUM_GPUS:-8}

TRAIN_FILES=/mnt/hdfs/zhangchi.usc1992_lf_lq/data/genselect_dataset.parquet

backend=${BACKEND:-fsdp}

project_name=verl_sft_test

RESUME_MODE=auto

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

MODEL_PATH=/mnt/hdfs/zhangchi.usc1992_lf_lq/models/Qwen3-8B-Base

SP_SIZE=${SP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:-${NUM_GPUS}}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE}"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=nvidia-openmathreasoning-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp${FSDP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=nvidia-openmathreasoning-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}
fi

ckpts_home=${ckpts_home:-/mnt/hdfs/zhangchi.usc1992_ssd_hldy/open_verl/sft/${project_name}/${exp_name}}
mkdir -p "${ckpts_home}"

/opt/tiger/internal/verl/verl/tools/internal/TORCHRUN ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=16 \
    data.max_length=16384 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=16384 \
    data.messages_key=messages \
    model.path=$MODEL_PATH \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=2000 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=2 \
    checkpoint.save_contents=[model,optimizer,extra] \

    # trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    
    