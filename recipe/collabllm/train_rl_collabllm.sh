# Usage: sh recipe/collabllm/train_rl_collabllm.sh <optional resume path>

set -x

PROJECT_DIR="$(pwd)"
export VLLM_USE_V1=1

RESUME_PATH="${1:-}"

if [ -z "$RESUME_PATH" ]; then
    RESUME_PATH=null
fi

DATASET=math-hard-large
PROJECT_DIR="$(pwd)"
AGENTLOOP_CONFIG_PATH="$PROJECT_DIR/recipe/collabllm/config/agent.yaml"


python3 -m verl.trainer.main_ppo \
    trainer.val_before_train=False \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/collabllm-$DATASET/rl_train.parquet \
    data.val_files=$HOME/data/collabllm-$DATASET/rl_validation.parquet \
    reward_model.reward_manager=collabllm \
    +reward_model.reward_kwargs.metric_weights.accuracy=1 \
    +reward_model.reward_kwargs.metric_weights.interactivity=1 \
    +reward_model.reward_kwargs.metric_weights.token_amount=-0.0001 \
    +reward_model.reward_kwargs.llm_judge_kwargs.model=gpt-4o-mini \
    +reward_model.reward_kwargs.llm_judge_kwargs.max_tokens=2048 \
    +reward_model.reward_kwargs.llm_judge_kwargs.temperature=0 \
    data.train_batch_size=16 \
    data.max_prompt_length=8196 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts=3 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENTLOOP_CONFIG_PATH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=verlxcollabllm \
    trainer.experiment_name=collabllm-qwen2.5-7B-$DATASET \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    custom_reward_function.path=recipe/collabllm/reward_function.py \
    custom_reward_function.name=conversation_level_reward_func \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/recipe/collabllm/config/collabllm_interaction_config.yaml" \
    trainer.resume_from_path=$RESUME_PATH 
