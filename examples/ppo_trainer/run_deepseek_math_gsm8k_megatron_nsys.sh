set -x

# Example runnable on H20 * 8

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files=${train_files:-"$gsm8k_train_path"}
test_files=${test_files:-"$gsm8k_test_path"}

# Nsight profiling configuration
PROFILE_STEPS="[1,2,5]" # or [] or null
PROFILE_RANKS_ALL=False # or True
PROFILE_RANKS=[0,4,8,12]
DISCRETE=True  # or True

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.profiler.ranks=$PROFILE_RANKS \
    actor_rollout_ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.profiler.discrete=$DISCRETE \
    critic.optim.lr=1e-5 \
    critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.profiler.ranks=$PROFILE_RANKS \
    critic.profiler.all_ranks=$PROFILE_RANKS_ALL \
    critic.profiler.discrete=$DISCRETE \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_ppo_gsm8k_math_examples' \
    trainer.experiment_name='deepseek_llm_7b_megatron' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=6 \
    trainer.profile_steps=$PROFILE_STEPS $@
