#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NUM_GPUS}}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp"}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}
PAD_MODE=${PAD_MODE:-no_padding}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
LR="1e-5"
MINLR="1e-6"

export VERL_SFT_LOGGING_LEVEL=INFO
#pip install --upgrade mbridge==0.15.1 megatron-core==0.14

#pip install --upgrade mbridge==0.15.1 megatron-core==0.13

mode=${mode:-spmd}
backend=${BACKEND:-megatron}

# git clone https://github.com/ArronHZG/mbridge/tree/feature/verl_mtp -b feature/verl_mtp
PYPATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yumingxuan/pythonlib
export PYTHONPATH=$PYTHONPATH:$PYPATH/mbridge:$PYPATH/Megatron-LM

TENSORBOARD_DIR=

if [ "$mode" = "spmd" ]; then
  MASTER_ADDR=${MASTER_ADDR:-localhost}
  MASTER_PORT=${MASTER_PORT:-29500}
  NNODES=${NNODES:-1}
  RANK=${RANK:-0}
  
  ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

  COMMAND="torchrun --nnodes=$NNODES --nproc-per-node=${NUM_GPUS} --node-rank=$RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT ${ENTRYPOINT}"

  #COMMAND="python -m debugpy --listen 8414 --wait-for-client -m torch.distributed.run --nnodes=$NNODES --nproc-per-node=${NUM_GPUS} --node-rank=$RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT ${ENTRYPOINT}"
  #export TENSORBOARD_DIR=/tmp/dummy_tb
else
  # Ray mode - fallback for compatibility
  ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer_ray"}
  COMMAND="python -X faulthandler ${ENTRYPOINT} trainer.nnodes=${NNODES:-1} trainer.n_gpus_per_node=${NUM_GPUS:-8}"
fi

DATASET_DIR=${DATASET_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yumingxuan/dataset/rl/gsm8k_full_prompt}
TRAIN_FILES=${DATASET_DIR}/train.parquet
VAL_FILES=${DATASET_DIR}/test.parquet

project_name=verl_sft_test

RESUME_MODE=disable

ckpts_home=${ckpts_home:-~/verl/test/gsm8k-sft-${backend}}
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/deepsearch_files_ssd/LLMbasemodels/huggingface.co/XiaomiMiMo/MiMo-7B-RL
# initial setup
if [ ! -d "/workdir/mimo" ]; then
    #pip uninstall -y xformers.  # useless as install as root
    rsync -rPv "$MODEL_PATH"/ /workdir/mimo
    pip install --upgrade mbridge==0.15.1 qwen_vl_utils
    python3 -c 'import megatron.core; megatron_version = getattr(megatron.core, "__version__"); print(f"megatron_version is {megatron_version}", megatron_version >= "0.14")'
fi

export HF_HOME="/tmp/hf_home_mimo"
export PYTHONPATH="${PYTHONPATH}:/tmp/hf_home_mimo/modules/"

MODEL_PATH="/workdir/mimo"


#PAD_MODE=${PAD_MODE:-left_right}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    +optim.lr_warmup_steps=10 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"

VEOMNI_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_min=1e-6 \
    optim.lr_scheduler_type=cosine \
    engine.ulysses_parallel_size=${SP_SIZE} \
    engine.data_parallel_mode=${FSDP_STRATEGY} \
    engine.data_parallel_size=${FSDP_SIZE}"

   #+engine.override_transformer_config.num_layers_in_first_pipeline_stage=6 \
   #+engine.override_transformer_config.num_layers_in_last_pipeline_stage=5 \

MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=${LR} \
    optim.min_lr=${MINLR} \
    optim.lr_warmup_steps=10 \
    optim.weight_decay=0.1 \
    optim.betas='[0.9,0.95]' \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    engine.override_transformer_config.recompute_method=uniform \
    engine.override_transformer_config.recompute_granularity=full \
    engine.override_transformer_config.recompute_num_layers=1 \
    engine.use_dist_checkpointing=False \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    "

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=gsm8k-${backend}-${FSDP_STRATEGY}-sp${SP_SIZE}-fsdp${FSDP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}-mode-${mode}
elif [ "$backend" = "veomni" ]; then
    ENGINE_CONFIG="$VEOMNI_ENGINE_CONFIG"
    echo "Using veomni engine"
    exp_name=gsm8k-${backend}-sp${SP_SIZE}-fsdp${FSDP_SIZE}-pad-${PAD_MODE}-use_remove_padding-${USE_REMOVE_PADDING}-mode-${mode}
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=gsm8k-${backend}-tp${TP_SIZE}-pp${PP_SIZE}-vpp${VPP_SIZE}-cp${CP_SIZE}-lr-${MINLR}-${LR}
fi

date=$(date +"%Y%m%d_%H%M")
[ -z "$TENSORBOARD_DIR" ] && export TENSORBOARD_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yumingxuan/tensorboard/verl_sft_tensorboard_mimo/${exp_name}/${date}
mkdir -p $TENSORBOARD_DIR
export LOGDIR=$TENSORBOARD_DIR/output.log
export ckpts_home=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/yumingxuan/model/verl_sft_tensorboard_mimo/${exp_name}/${date}

mkdir -p "${ckpts_home}"

$COMMAND \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TRAIN_FILES}" \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.max_length=1024 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=prompt \
    data.num_workers=0 \
    model.path=$MODEL_PATH \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    model.trust_remote_code=True \
    model.mtp.enable=True \
    ${ENGINE_CONFIG} \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.resume_mode=${RESUME_MODE} 2>&1 | tee -a "${LOGDIR}"

    # trainer.total_training_steps=${TOTAL_TRAIN_STEP} \
    # trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    # trainer.max_ckpt_to_keep=1 \
    
#rm -rf "${ckpts_home:?}/*"