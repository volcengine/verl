#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

NUM_GPUS=${NUM_GPUS:-1}

TRAIN_FILES=libero_dataset
VAL_FILES=libero_dataset

backend=${BACKEND:-fsdp}

project_name=verl_sft_test

RESUME_MODE=disable

ckpts_home="checkpoints/"

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH="/file_system/common-models/Haozhan72-kangsheng/Openvla-oft-SFT-libero10-traj1"
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NUM_GPUS}}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp"}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-4 \
    optim.lr_warmup_steps_ratio=0.02 \
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
    exp_name=gsm8k-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp${FSDP_SIZE}
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=gsm8k-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}
fi

mkdir -p "${ckpts_home}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=16 \
    data.max_length=8192 \
    data.pad_mode=left_right \
    data.truncation=error \
    data.use_dynamic_bsz=False \
    data.max_token_len_per_gpu=20480 \
    data.messages_key=messages \
    data.pad_mode=right \
    model.path=$MODEL_PATH \
    model.trust_remote_code=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=2000000 \
    trainer.save_freq=3000 \
    trainer.logger=['console','file'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10000 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} \

    # trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    # trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    # trainer.max_ckpt_to_keep=1 \
    
# rm -rf "${ckpts_home:?}/*"