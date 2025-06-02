set -x
HYDRA_FULL_ERROR=1
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

TRAIN_FILES=$HOME/data/subtask_direct_distill/mix/train/executor.parquet # "[\"$HOME/data/subtask_direct_distill/mix/train/executor.parquet\", \"$HOME/data/subtask_direct_distill/mix/train/reward_model.parquet\"]"
VAL_FILES=$HOME/data/subtask_direct_distill/mix/test/executor.parquet # "[\"$HOME/data/subtask_direct_distill/mix/test/executor.parquet\", \"$HOME/data/subtask_direct_distill/mix/test/reward_model.parquet\"]"

REWARD_FILE=orby/reward/subtask.py
REWARD_FN=reward_func

echo "If you encounter OOM, try tweaking the following parameters:"
echo "data.train_batch_size"
echo "actor_rollout_ref.actor.ppo_mini_batch_size"
echo "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
echo "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
echo "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
echo "actor_rollout_ref.rollout.n"

python3 -m orby.trainer.main_ppo \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=32 \
    +data.max_prompt_length=7680 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    +actor_rollout_ref.rollout.limit_images=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_subtask' \
    trainer.experiment_name='qwen2_5_vl_7b_subtask' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=2 $@
