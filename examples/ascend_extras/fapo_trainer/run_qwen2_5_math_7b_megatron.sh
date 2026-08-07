#!/usr/bin/env bash
# FAPO on Ascend: Qwen2.5-Math-7B actor (Megatron) + FAPO-GenRM-4B reward model.
#
# Both roles share one 8-device node: the actor/ref take ACTOR_GPUS devices and
# the generative RM takes a separate resource pool of RM_GPUS devices. Set
# REWARD_MODE=remote to drive an externally launched GRM service instead.
#
# Platform and inference backend are selected through DEVICE / INFER_BACKEND so
# this file stays free of hardware-specific names (tests/special_sanity/
# check_example_naming.py rejects vllm/sglang/npu/fp8 in example filenames).

set -xeuo pipefail

project_name='FAPO-Ascend'
exp_name='FAPO-Qwen2.5-Math-7B-megatron'

########################### user-adjustable ###########################
DEVICE=${DEVICE:-npu}
INFER_BACKEND=${INFER_BACKEND:-vllm}
REWARD_MODE=${REWARD_MODE:-colocate}          # colocate | remote

RAY_DATA_HOME=${RAY_DATA_HOME:-"/data/user"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen/Qwen2.5-Math-7B"}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen/Qwen2.5-Math-7B-mcore"}
GRM_PATH=${GRM_PATH:-"${RAY_DATA_HOME}/models/FAPO-GenRM-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/fapo/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/fapo/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# Device split. The RM runs in its own resource pool on the same node.
TOTAL_GPUS=${TOTAL_GPUS:-8}
RM_GPUS=${RM_GPUS:-2}
ACTOR_GPUS=${ACTOR_GPUS:-$((TOTAL_GPUS - RM_GPUS))}

# Sequence budget. 3k of response is a floor, not a tuning knob: FAPO's reward
# depends on the GRM spotting flawed-but-correct reasoning, and shorter rollouts
# stop producing those, so trade batch size away before touching this.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}

# Megatron shape for the actor. TP is bounded by GQA: Qwen2.5-Math-7B has 28
# heads over 4 KV heads, so TP must divide 4 and also divide ACTOR_GPUS.
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=${TRAIN_PP:-1}
TRAIN_CP=${TRAIN_CP:-1}
INFER_TP=${INFER_TP:-2}

# Dynamic batching is what keeps the static divisibility assert in
# verl/utils/config.py:95-113 out of the picture: with ACTOR_GPUS=6 and
# TP*PP*CP=2 the data-parallel size is 3, and train_batch_size*n = 128 has no
# factor of 3, so every static micro-batch choice would be rejected.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))}
INFER_MAX_TOKEN_LEN=${INFER_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-25}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# The shipped dataset repeats DAPO-Math-17k twenty times (347,960 rows) to feed
# 200 steps at batch 512. This run needs TRAIN_BATCH_SIZE*TOTAL_TRAINING_STEPS
# prompts, so cap the dataset instead of filtering rows that will never be read.
# rl_dataset.py applies the cap before filtering, so this also bounds the filter.
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-$((TRAIN_BATCH_SIZE * TOTAL_TRAINING_STEPS * 4))}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}

# trainer/config/data/legacy_data.yaml pins filter_overlong_prompts_workers to 1
# while the code default is cpu_count()//4. Filtering runs to completion before
# any device work starts, so the single-process default is idle accelerator time.
FILTER_WORKERS=${FILTER_WORKERS:-$(python3 -c 'import os; print(max(1, os.cpu_count() // 4))' 2>/dev/null || echo 8)}

# FAPO algorithm knobs, mirroring recipe/fapo/run_fapo_7b.sh.
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
# Megatron keeps gradient clipping on the optimizer (config/optim/megatron.yaml),
# not on the actor as the FSDP tree does, so the reference script's
# actor.grad_clip would be rejected here.
GRAD_CLIP=${GRAD_CLIP:-1.0}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
# overlong_buffer.len must not exceed max_resp_len or DAPORewardManager asserts
# at start-up (reward_loop/reward_manager/dapo.py:39-50). The reference uses
# 4096 against an 8k response budget; scale it with the response length here.
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 3))}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

GRM_MEM_FRACTION=${GRM_MEM_FRACTION:-0.90}
GRM_TP=${GRM_TP:-1}
GRM_MAX_MODEL_LEN=${GRM_MAX_MODEL_LEN:-8192}
GRM_ENFORCE_EAGER=${GRM_ENFORCE_EAGER:-False}
ROUTER_ADDRESS=${ROUTER_ADDRESS:-""}          # REWARD_MODE=remote only
########################### end user-adjustable ###########################

# HuggingFace caches default to /root/.cache, which lives on the container's
# overlay filesystem and disappears with the pod. Every launch would then re-download
# and re-filter. Park them on the shared filesystem alongside the data instead.
export HF_HOME=${HF_HOME:-"${RAY_DATA_HOME}/hf_cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}
# Not fatal: an unwritable cache path costs a re-download, not a failed run, and
# making it fatal would also break config-only dry runs off the shared filesystem.
mkdir -p "${HF_DATASETS_CACHE}" 2>/dev/null || echo "warning: cannot create ${HF_DATASETS_CACHE}, HF cache will not persist" >&2

# Dataset loading and prompt filtering happen inside a Ray actor. A locally
# started Ray inherits this shell's environment, but a job submitted to an
# existing cluster does not, and get_ppo_ray_runtime_env() only forwards a fixed
# allowlist. Forward the cache location explicitly so both paths behave the same.
RAY_ENV_ARGS=(
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="${HF_HOME}"
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
)

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

if [ "${REWARD_MODE}" = "remote" ] && [ -z "${ROUTER_ADDRESS}" ]; then
    echo "REWARD_MODE=remote requires ROUTER_ADDRESS to point at the GRM router" >&2
    exit 1
fi

# Reward wiring. verl.trainer.main_ppo reads config.reward.*; the top-level
# reward_model.* keys that recipe/fapo/run_fapo_7b.sh uses are only migrated by
# the fully-async entrypoint, so spelling them the old way here would be accepted
# by hydra and then silently ignored.
REWARD_ARGS=(
    reward.reward_manager.name=dapo
    reward.custom_reward_function.name=compute_score_fapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}"
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}"
    +reward.reward_kwargs.overlong_buffer_cfg.log=True
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
)

if [ "${REWARD_MODE}" = "colocate" ]; then
    REWARD_ARGS+=(
        reward.custom_reward_function.path=recipe/fapo/reward_fn_reasoning.py
        reward.reward_model.enable=True
        reward.reward_model.enable_resource_pool=True
        reward.reward_model.n_gpus_per_node="${RM_GPUS}"
        reward.reward_model.nnodes=1
        reward.reward_model.model_path="${GRM_PATH}"
        reward.reward_model.rollout.name="${INFER_BACKEND}"
        # reward/reward.yaml ships tp=2, util=0.5, enforce_eager=true and
        # free_cache_engine=true; all four are wrong for a resident 4B judge.
        reward.reward_model.rollout.tensor_model_parallel_size="${GRM_TP}"
        reward.reward_model.rollout.gpu_memory_utilization="${GRM_MEM_FRACTION}"
        reward.reward_model.rollout.enforce_eager="${GRM_ENFORCE_EAGER}"
        reward.reward_model.rollout.free_cache_engine=False
        reward.reward_model.rollout.max_model_len="${GRM_MAX_MODEL_LEN}"
    )
else
    REWARD_ARGS+=(
        reward.custom_reward_function.path=recipe/fapo/reward_fn_reasoning_remote.py
        reward.reward_model.enable=False
        reward.reward_model.enable_resource_pool=False
        +reward.custom_reward_function.reward_kwargs.reward_router_address="${ROUTER_ADDRESS}"
    )
fi

python3 -m verl.trainer.main_ppo --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers="${FILTER_WORKERS}" \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}" \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad="${GRAD_CLIP}" \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.name="${INFER_BACKEND}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    "${REWARD_ARGS[@]}" \
    "${RAY_ENV_ARGS[@]}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.device="${DEVICE}" \
    trainer.n_gpus_per_node="${ACTOR_GPUS}" \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
