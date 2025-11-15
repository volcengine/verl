set -x

project_name='GRPO'
exp_name='DeepSeekV3-671B-GRPO-Megatron-256rank-gbs512'

NNODES=16
NPUS_PER_NODE=16

adv_estimator=grpo

kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))
max_num_batched_tokens=1024
ppo_mini_batch_size=512

train_prompt_bsz=512
n_resp_per_prompt=16

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CONFIG_PATH=${CONFIG_PATH:-"${RAY_DATA_HOME}/verl/trainer/config"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/DeepSeek-V3-hf"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/DeepseekV3-dist-ckpts"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/deepscaler/test.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
offload=True
max_num_seqs=64
gen_tp=2

# Currently, it is necessary to enable `enable_chunked_prefill` in the script. 
# However, in vLLM ascend, this configuration is off by default and does not take effect.
python3 -m recipe.r1_ascend.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='ppo_megatron_trainer.yaml' \
    custom_reward_function.path=recipe/r1_ascend/deepscaler.py \
    custom_reward_function.name=compute_score \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}  \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=32 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=8 \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${CKPTS_DIR}" \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${CKPTS_DIR}" \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.device="npu" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.multi_head_latent_attention=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers=[[6],[8],[8],[8],[8],[8],[8],[7]] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.rope_scaling_type='yarn' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.yarn_scaling_factor=40 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.rope_scaling_mscale=1.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.rope_scaling_mscale_all_dim=1.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=6 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=7 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True $@