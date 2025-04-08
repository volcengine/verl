set -x

# the config file used: verl/trainer/main_ppo/config/ppo_megatron_trainer.yaml

mkdir -p $HOME/models

huggingface-cli download deepseek-ai/deepseek-coder-1.3b-instruct --local-dir $HOME/models/deepseek-ai/deepseek-coder-1.3b-instruct

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$HOME/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    critic.optim.lr=2e-5 \
    critic.model.path=$HOME/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.megatron.pipeline_model_parallel_size=2 \
    critic.megatron.virtual_pipeline_model_parallel_size=2 \
    critic.megatron.tensor_model_parallel_size=2 \
    critic.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='deepseek_megatron_checkpoint_saveload' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1 $@


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$HOME/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    critic.optim.lr=2e-5 \
    critic.model.path=$HOME/models/deepseek-ai/deepseek-coder-1.3b-instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.megatron.pipeline_model_parallel_size=2 \
    critic.megatron.virtual_pipeline_model_parallel_size=2 \
    critic.megatron.tensor_model_parallel_size=2 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='deepseek_megatron_checkpoint_saveload' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1 $@