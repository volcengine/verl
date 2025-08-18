set -x
logs=/home/l00878165/sglang/repos/logs/tmp61.log
export ASCEND_LAUNCH_BLOCKING=1

# profiling configuration
PROFILE_STEPS="[1]"
PROFILE_RANKS_ALL=False
DISCRETE=True
PROFILE_RANKS="[0,1,2]"

# profiling NPU options
SAVE_PATH="/home/l00878165/sglang/repos/logs/profiling/close"
LEVEL="level1"
WITH_MEMORY=True
RECORD_SHAPES=False
WITH_NPU=True
WITH_CPU=True
WITH_MODULE=False
WITH_STACK=True
ANALYSIS=True
ROLES=["all"]


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/l00878165/datasets/processed_gsm8k/train.parquet \
    data.val_files=/home/l00878165/datasets/processed_gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/l00878165/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_5_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10000 \
    trainer.total_epochs=5 \
    trainer.device=npu  2>&1 | tee -i $logs