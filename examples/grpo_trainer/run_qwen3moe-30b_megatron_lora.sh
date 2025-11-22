TP=2
PP=2
CP=2
EP=4
ETP=1


# megatron tuning guide:
# 1. recommend to offload all states by setting ALL_OFFLOAD=True
# 2. enable dynamic batch size by setting actor_rollout_ref.actor.use_dynamic_bsz=True ref.log_prob_use_dynamic_bsz=True rollout.log_prob_use_dynamic_bsz=True
# 3. set ppo_max_token_len_per_gpu and log_prob_max_token_len_per_gpu as large as possible for better MFU (limited by GPU memory). assure ppo_max_token_len_per_gpu > max_prompt_length+max_response_length, if sequence length is too long, you can increase the TP/PP size
# 4. if memory is very limited, enable full recompute, but the mfu will be 30% lower
#        full recompute settings:
#        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
#        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
#        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

# Need to install Megatron-Bridge
# NOTE: Make sure you use Megatron-Bridge later than 0.2.0 
# (after https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/36302b7ca1305f0690e17cf4e4019ac822746872)
# for MoE LoRA When you want to set ETP and ETP!=TP.
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1363


ALL_OFFLOAD=${ALL_OFFLOAD:-False}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.truncation='error' \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    actor_rollout_ref.model.lora.rank=16 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.ref.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen3_30b_a3b_megatron_lora' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
