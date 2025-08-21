set -x


# +reward_model.reward_kwargs.metric_weights.accuracy=1 \
# +reward_model.reward_kwargs.metric_weights.interactivity=1 \
# +reward_model.reward_kwargs.metric_weights.token_amount=-0.0005 \

PROJECT_DIR="$(pwd)"
set -x


PROJECT_DIR="$(pwd)"
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    --config-dir recipe/collabllm/config \
    --config-name collabllm_trainer \
    trainer.val_before_train=False \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/collabllm-math-hard/train.parquet \
    data.val_files=$HOME/data/collabllm-math-hard/validation.parquet \
    reward_model.reward_manager=collabllm \
    +reward_model.reward_kwargs.metric_weights.accuracy=1 \
    +reward_model.reward_kwargs.metric_weights.interactivity=1 \
    +reward_model.reward_kwargs.metric_weights.token_amount=-0.0005 \
    +reward_model.reward_kwargs.llm_judge_kwargs.model=claude-3-5-sonnet-latest \
    +reward_model.reward_kwargs.llm_judge_kwargs.temperature=0 \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verlxcollabllm' \
    trainer.experiment_name='collabllm' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    custom_reward_function.path=recipe/collabllm/reward_function.py \
    custom_reward_function.name=conversation_level_reward_func \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/recipe/collabllm/config/collabllm_interaction_config.yaml" 