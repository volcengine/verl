#!/bin/bash

## Instruction
# set your env 
#   DATASET_DIR : base of dataset directory
#   WANDB_API_KEY : your wandb api key
#   OUTPUT_BASEPATH : your output directory
#   /workspace/models : create soft links to /workspace/models

# source nccl_env.sh

# or you can use lmsys/gpt-oss-20b-bf16
# recommend to use same value for train_batch_size and ppo_mini_batch_size
# to avoid MOE training instability
# use large value for max_response_length if you want to use reasoning effort high.
HF_MODEL_NAME="gpt-oss-20b-BF16"

# Megatron distribued ckpt
MCORE_MODEL_NAME="gpt-oss-20b-BF16-to-mcore_bridge-tp8-pp1-cp1-ep8-bf16/iter_0000000"

export WANDB_PROJECT=gpt-oss
export WAND_EXP_NAME=gpt-oss-megatron-test-run
export WANDB_API_KEY=${WANDB_API_KEY}

MODEL_DIR="/workspace/models"

MEGATRON_DIST_CKPT_MODEL_DIR="${MODEL_DIR}/${MCORE_MODEL_NAME}"

HF_MODEL_PATH="${MODEL_DIR}/${HF_MODEL_NAME}"

gsm8k_train_path=${DATASET_DIR}/gsm8k/train.parquet
gsm8k_test_path=${DATASET_DIR}/gsm8k/test.parquet

OUTPUT_BASEPATH=${OUTPUT_BASEPATH}

CKPT_DIR="${OUTPUT_BASEPATH}/checkpoint/"

mkdir -p $CKPT_DIR 

# 1x8 H100/H800 (80GB)
NODES=1
PP=1
TP=8
EP=8
ETP=1

offload=True

## model params

max_prompt_length=512 # $((1024 * 2)) # 512
max_response_length=4096 # $((1024 * 4)) # 4096

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))

infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1))

## Data
train_batch_size=32 # 256

#     data.truncation='left' \
DATA=" 
    data.train_files='$gsm8k_train_path' \
    data.val_files='$gsm8k_test_path' \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.return_raw_chat=False \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.reasoning_effort=medium \
"

## Opt
MEGATRON_OPT="
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
"

ACTOR_OPTIM_OPT="
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
"

OPT="
    actor_rollout_ref.model.use_fused_kernels=True \
    $MEGATRON_OPT \
    $ACTOR_OPTIM_OPT \
"

## Actor
clip_ratio_low=0.2
clip_ratio_high=0.28

use_dynamic_bsz=False
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=1

ACTOR_GRAD_PARAM_OFFLOAD="
    actor_rollout_ref.actor.megatron.grad_offload=True \
"

ACTOR_PARAM_OFFLOAD="
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    $ACTOR_GRAD_PARAM_OFFLOAD \
"

ACTOR_REF_PARAM_OFFLOAD="
    actor_rollout_ref.ref.megatron.param_offload=True \
"

ACTOR_PARALLEL="
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
"


ACTOR_REF_PARALLEL="
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
"


ACTOR="
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    ${ACTOR_PARAM_OFFLOAD} \
    ${ACTOR_REF_PARAM_OFFLOAD} \
    ${ACTOR_PARALLEL} \
    ${ACTOR_REF_PARALLEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=\"token-mean\" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
"


## Rollout engine parameters
rollout_dtype="bfloat16" # mxfp4
gptoss_rollout_tp_size=8
memory_usage_rage=0.5

temperature=1.
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM/SGLang rollout
val_top_p=0.7

n_resp_per_prompt=16

micro_batch_size_per_gpu=8

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=sync
    actor_rollout_ref.rollout.dtype=${rollout_dtype}
    actor_rollout_ref.rollout.gpu_memory_utilization=$memory_usage_rage
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gptoss_rollout_tp_size
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

ROLLOUT_REF="
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
"

export VERL_LOGGING_LEVEL=INFO

export PYTHONPATH=/home/yiakwy/workspace/Github/sglang/3rdparty/Megatron-LM:$PYTHONPATH

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${HF_MODEL_PATH}" \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MEGATRON_DIST_CKPT_MODEL_DIR} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MEGATRON_DIST_CKPT_MODEL_DIR} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    $DATA \
    "${ROLLOUT[@]}" \
    $ROLLOUT_REF \
    $ACTOR \
    $OPT \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='oai_oss_20b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.val_before_train=0 \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.resume_mode=auto \
    trainer.total_epochs=15 $@ 2>&1 | tee verl_megatron.log
