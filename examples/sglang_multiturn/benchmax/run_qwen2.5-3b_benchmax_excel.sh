# run on 8xH20
# make sure your current working directory is the root of the project
# Specifically note the last 3 lines
# The first line points to tool config, which is necessary for initializing tools from the benchmax environment
# The second and third lines point to the relevant benchmax environment to initialize the rewards from

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"


TRAIN_DATA="/data/excel/train.parquet"
VAL_DATA="/data/excel/test.parquet"

TOOL_CONFIG="$CONFIG_PATH/tool_config/benchmax_excel_tool_config.yaml"
BENCHMAX_CLASS_NAME="benchmax.envs.excel.excel_env.ExcelEnv"


PYTHONPATH="$PYTHONPATH:$(pwd)" python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='benchmax_excel_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=10000 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='wiki_search' \
    trainer.experiment_name='qwen2.5-3b-instruct_wiki_search' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    trainer.total_epochs=10 $@ \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    reward_model.reward_manager=benchmax \
    +reward_model.reward_kwargs.benchmax_cls_name="$BENCHMAX_CLASS_NAME"