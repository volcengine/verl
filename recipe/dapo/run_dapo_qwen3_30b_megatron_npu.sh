#!/usr/bin/env bash
# git apply /sfs_turbo/wlf/VerlCode/docker/verl-npu/code/verl_dense.patch
cp -f /sfs_turbo/wlf/VerlCode/dev/skip_infer/verl/recipe/dapo/config/dapo_trainer-fsdp.yaml /opt/verl/recipe/dapo/config/dapo_trainer-fsdp.yaml
cp -f /sfs_turbo/wlf/VerlCode/dev/skip_infer/verl/recipe/dapo/config/dapo_trainer-megatron.yaml /opt/verl/recipe/dapo/config/dapo_trainer-megatron.yaml
cp -f /sfs_turbo/wlf/stable/verl-dapo/verl/trainer/ant_runtime_env.yaml /opt/verl/verl/trainer/ant_runtime_env.yaml

if [ $(pip list | grep vllm-ascend | wc | awk '{print $1}') != 1 ];then
    cd /opt/vllm-ascend
    pip install -e . -v --no-deps
fi


cd /opt/verl

project_name='DAPO'
exp_name='DAPO-qwen3-30b-megatron'
NNODES=${NNODES:-1}
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
li=2
lo=34
max_prompt_length=$((1024 * li))
max_response_length=$((1024 * lo))
#max_prompt_length=1024
#max_response_length=2048
enable_overlong_buffer=False
overlong_buffer_len=$((4 * 1024))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
enable_filter_groups=True
#enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=8 #768
gen_prompt_bsz=8
# gen_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=8

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# Paths
#MODEL_PATH=${MODEL_PATH:-"/sfs_turbo/zzy/model/Qwen2.5-32B-Instruct"}


MCORE_MODEL_PATH="/sfs_turbo/mcore/Qwen3-30B-A3B"
MODEL_PATH="/sfs_turbo/pretrained_models/Qwen3-30B-A3B"
#TRAIN_FILE="/sfs_turbo/qinzheng.lz/tokenized_data/dapo-math-17k-update-reasoning.parquet"
#TRAIN_FILE="/sfs_turbo/wlf/LocalCode/dataset/dapo-math-17k.parquet"
TRAIN_FILE=/sfs_turbo/wlf/VerlCode/dataset/dapo-math-17k-shuffled-myc.parquet
TEST_FILE="/sfs_turbo/zzy/data/test.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7


# Performance Related Parameter
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

max_num_batched_tokens=$((actor_ppo_max_token_len))
#actor_ppo_max_token_len=32768
#infer_ppo_max_token_len=32768

offload=True
# gen_tp=2
# gen_dp=2
gen_tp=4
gen_dp=1
gen_world_size=$((NNODES * 16))

train_tp=4
train_ep=2
train_pp=2
train_cp=1


RUNTIME_ENV=verl/trainer/mc2_env.yaml
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    python3 -m recipe.dapo.main_dapo \
    --config-name="dapo_trainer-megatron" \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.filter_overlong_prompts=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.shuffle=True \
    data.prompt_key=prompt \
    actor_rollout_ref.rollout.name=vllm \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
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
    actor_rollout_ref.actor.ppo_epochs=1 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    +actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.critic.optim.lr=5e-8 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    +actor_rollout_ref.rollout.dp_model_parallel_size=${gen_dp} \
    +actor_rollout_ref.rollout.rollout_world_size=${gen_world_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    +actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="/sfs_turbo/myc_mc2/xcxcxxccx" \
    trainer.device="npu" \
    actor_rollout_ref.nccl_timeout=14400 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    ++actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \

    # actor_rollout_ref.rollout.enforce_eager=True \

