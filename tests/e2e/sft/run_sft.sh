#!/usr/bin/env bash
set -x

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.fsdp_sft_trainer"}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

CKPTS_HOME=${CKPTS_HOME:-$HOME/ckpts}

SP_SIZE=${SP_SIZE:-1}
LIGER=${LIGER:-False}
MULTITURN=${MULTITURN:-False}
LORA_RANK=${LORA_RANK:-0}
RM_PAD=${RM_PAD:-True}

exp_name="$(basename "${MODEL_ID,,}")-sft-minimal-$(git rev-parse --short HEAD)"

read -r -d '' cmd <<EOF
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    "${ENTRYPOINT}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.multiturn.enable="${MULTITURN}" \
    data.multiturn.messages_key=messages \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain="${MODEL_ID}" \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha=16 \
    model.target_modules=all-linear
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_liger="${LIGER}" \
    use_remove_padding="${RM_PAD}" \
    trainer.default_local_dir="${CKPTS_HOME}" \
    trainer.project_name="verl-test" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_training_steps="${TOT_TRAIN_STEPS}" \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@
EOF

eval "$cmd"

rm -rf "${CKPTS_HOME}"