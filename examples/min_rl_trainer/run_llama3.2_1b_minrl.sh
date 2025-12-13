#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# High-level experiment configuration
###############################################################################

MODEL_PATH=""

DATA_DIR=""
CKPTS_DIR=""

GPUS_PER_NODE=8
NNODES=1

###############################################################################
# Sequence lengths and token budgets
###############################################################################

max_prompt_length=512
max_response_length=$((8192 - max_prompt_length))   # 7680

actor_ppo_max_token_len=8192
infer_ppo_max_token_len=8192

###############################################################################
# GRPO + MiniRL policy loss
###############################################################################

# Group-normalized outcome reward (same as MiniRL's Â).
adv_estimator="grpo"

# Your custom policy loss implementing MiniRL-style clipping/gating
# and supporting rollout_is_weights.
loss_mode="minirl"

# Number of responses per prompt (group size).
n_resp_per_prompt=16

# Global batch in *prompts* (before multiplying by n_resp_per_prompt).
train_batch_size=512

# PPO inner-loop batch/mini-batch config (in prompts).
ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=4

# Recommended for sequence-level losses & MiniRL-style gating.
loss_agg_mode="seq-mean-token-mean"

###############################################################################
# vLLM rollout / sampling
###############################################################################

rollout_engine="vllm"
rollout_mode="async"

gpu_memory_utilization=0.7
gen_tp=1

temperature=1.0
top_p=1.0
top_k=-1

val_top_p=0.7

###############################################################################
# KL (off for pure RLVR-style setup)
###############################################################################

use_kl_in_reward=false
kl_coef=0.0

use_kl_loss=false
kl_loss_coef=0.0

###############################################################################
# Parallelism / FSDP
###############################################################################

offload=false
sp_size=1
use_dynamic_bsz=true

###############################################################################
# DAPO length penalty (optional)
###############################################################################

reward_manager="dapo"

enable_overlong_buffer=false
overlong_buffer_len=3584
overlong_penalty_factor=1.0

entropy_checkpointing=false

###############################################################################
# Training schedule / logging
###############################################################################

project_name="big-math-minirl"
exp_name="big-math-minirl-8kctx-mathstyle"

val_before_train=false
test_freq=5
save_freq=100
total_epochs=5

###############################################################################
# Rollout importance sampling & correction (decoupled mode)
###############################################################################
# These flags assume the newer rollout-IS API in VeRL. If your fork uses
# `algorithm.rollout_correction.*` instead, map them accordingly.

# Enable rollout IS correction (token-level).
rollout_is=true
rollout_is_level="token"          # "token" | "sequence"
rollout_is_mode="clip"            # "clip" | "rs" | "clip+rs" (depending on your version)
rollout_is_threshold=5.0          # truncate IS weights at this upper bound
rollout_is_batch_normalize=false   # normalize weights to mean ~ 1.0

# Optional: veto extremely large per-token weights even before clipping.
rollout_token_veto_threshold=10.0

###############################################################################
# Data paths
###############################################################################

train_files="$DATA_DIR/train.parquet"
test_files="$DATA_DIR/test.parquet"

###############################################################################
# Launch VeRL MiniRL-style training
###############################################################################

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    \
    algorithm.rollout_is=${rollout_is} \
    algorithm.rollout_is_level=${rollout_is_level} \
    algorithm.rollout_is_mode=${rollout_is_mode} \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2
