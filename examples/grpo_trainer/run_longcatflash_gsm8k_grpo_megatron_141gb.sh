#!/bin/bash
set -xeuo pipefail

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1
#export NCCL_P2P_DISABLE=1
#export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export PYTHONUNBUFFERED=1

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=60
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=15
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.7'
export TOKENIZERS_PARALLELISM=false
export VERL_LOGGING_LEVEL=INFO

RAY_DATA_HOME=${RAY_DATA_HOME:-"<you_data_home>"}
MODEL_PATH=$RAY_DATA_HOME/models/LongCatFlashChat
TRAIN_FILE=$RAY_DATA_HOME/data/gsm8k/train.parquet
TEST_FILE=$RAY_DATA_HOME/data/gsm8k/test.parquet

project_name='longcat-flash-chat-gsm8k'
exp_name="exp-01"
CKPTS_DIR=$RAY_DATA_HOME/ckpts/agent/${project_name}/${exp_name}
ROLLOUT_LOG_PATH=$RAY_DATA_HOME/logs/${project_name}/${exp_name}
timestamp=$(date +%s)
export TENSORBOARD_DIR=${RAY_DATA_HOME}/tensorboard/${project_name}-${timestamp}

# Algorithm
train_prompt_bsz=640
n_resp_per_prompt=4
train_prompt_mini_bsz=160

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=2
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
offload=True
optim_offload=${OFFLOAD_OPTIM:-True}
optim_cpu_offload=${OFFLOAD_OPTIM:-False}
optimizer_offload_fraction=${OFFLOAD_FRACTION:-null}

gen_tp=16
train_tp=${TP:-32}
train_pp=${PP:-4}
train_vpp=${PP:-1}
EP=${EP:-32}
ETP=1
CP=1
NNODES=$((train_tp * train_pp / 8))

unset http_proxy
unset https_proxy
which python3
which pip3
pip3 list
echo $PATH

# RAY_ADDRESS='auto' ray job submit --working-dir . --
python3 -X faulthandler -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${optim_cpu_offload} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${optim_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=5120 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.nccl_timeout=360000 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.ep_size=${gen_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0001 \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='fused' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules="['core_attn','mlp']" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=null \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=4 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=1 \
    trainer.rollout_data_dir="${ROLLOUT_LOG_PATH}"
