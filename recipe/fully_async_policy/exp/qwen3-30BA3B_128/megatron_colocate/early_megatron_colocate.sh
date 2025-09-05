#!/usr/bin/env bash
set -xeuo pipefail

project_name='DAPO'
exp_name='dapo_qwen3-30BA3B_32k_megatron_colocate_128_mbs32'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=512
train_prompt_mini_bsz=32
n_resp_per_prompt=16

NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/.friday/models/basemodel/Qwen/Qwen3-30B-A3B
CKPTS_DIR=./ckpts/${project_name}/${exp_name}
TRAIN_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/dapo-math-17k.parquet
TEST_FILE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-friday-studio/FTI/houzhenggang/data/dapo/aime-2024.parquet

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7


# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4
train_tp=1
train_pp=1
EP=8
ETP=1
CP=1

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=400 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules=["core_attn","moe_act","layernorm","mlp","moe"] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=False \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.use_mbridge=True

    # +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    # actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity="selective" \
    # actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules=["core_attn","moe_act","layernorm","mlp","moe"] \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=False \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=False \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=${last_layer} \