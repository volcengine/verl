set -x

# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
rollout_name="vllm"
chat_scheduler=examples.grpo_trainer.langgraph_chat_scheduler.LangGraphChatCompletionScheduler
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/grpo_trainer"

python3 -m verl.trainer.main_ppo \
    -cd $CONFIG_PATH \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files=$HOME/data/retool_dapo/train.parquet \
    data.val_files=$HOME/data/retool_aime2024/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    +actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    +config.actor_rollout_ref.rollout.multi_turn.enable=true \
    +config@actor_rollout_ref.rollout.langgraph=retool_reflection \
    +actor_rollout_ref.rollout.openai_serving_chat.enable_auto_tools=true \
    +actor_rollout_ref.rollout.openai_serving_chat.tool_parser=hermes \
    +actor_rollout_ref.rollout.enable_thinking=false \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='retool_langgraph' \
    trainer.experiment_name='qwen3-8b_langgraph_retool_sf' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.log_val_generations=32 \
    trainer.total_epochs=15 $@
