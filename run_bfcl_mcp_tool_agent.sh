set -x

ulimit -n 65535

export VLLM_USE_V1=1

# Include the following export if offline
export HF_HUB_OFFLINE=1
export MLFLOW_TRACKING_URI=""

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/tools/configs"
CONFIG_NAME="bfcl_multiturn_grpo"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=/data/user/minruixu/share/models/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","mlflow"]' \
    trainer.project_name='gsm8k_tool-agent' \
    trainer.experiment_name='qwen3-1.7b_function_rm-gsm8k-sgl-tool-agent-verify-n16' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_training_steps=2 \
    data.train_files=$PROJECT_DIR/data/bfcl/train.parquet \
    data.val_files=$PROJECT_DIR/data/bfcl/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/tools/mcp_configs/bfcl_mcp_server.json" \
    trainer.total_epochs=2 $@