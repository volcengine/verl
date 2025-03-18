#!/usr/bin/env bash
set -euxo pipefail

project_name='DAPO'
exp_name='DAPO-Qwen2.5-7B-Math-Test'

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RUNTIME_ENV=${RUNTIME_ENV:-"./verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-4}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
## Train
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))
gen_prompt_bsz=512
train_prompt_bsz=512
train_prompt_mini_bsz=32
n_resp_per_prompt=16
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout


# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${PWD}" \
    -- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.truncation='left' \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.25 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.fill_train_batch=True \
    algorithm.filter_groups.drop_last_mini_batch=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0\
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    custom_reward_function.overlong_buffer.enable=True \
    custom_reward_function.overlong_buffer.len=512 \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    +trainer.val_before_train=True \
    trainer.test_freq=2 \
    trainer.save_freq=2 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable