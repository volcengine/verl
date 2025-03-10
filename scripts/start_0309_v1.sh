set -x

# for debug
export HYDRA_FULL_ERROR=1

export WANDB_API_KEY=218cc9a2633c3b2303ca4dbc44397b10fa3e9115
export RAY_DEDUP_LOGS_ALLOW_REGEX="nodedup"
export VERL_PPO_LOGGING_LEVEL="INFO"

CONFIG_ARGS="
    --config-path=$(pwd)/configs \
    --config-name=qwen14b_sft_async \
"

DATASET_PREFIX=/workspace/lurui-yun/deep_research/prompts/res/hotpotQA_grok_system_harder_yst_23k_0306
WORLD_SIZE=$((MLP_WORKER_NUM * MLP_GPU))
BATCH_SIZE=$WORLD_SIZE

DATA_ARGS="
    data.train_files=$DATASET_PREFIX/train.parquet \
    data.val_files=$DATASET_PREFIX/test.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
"

ACTOR_ROLLOUT_REF_ARGS="
    actor_rollout_ref.actor.ppo_mini_batch_size=$WORLD_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE \
"

TRAINER_ARGS="
    trainer.n_gpus_per_node=$MLP_GPU \
    trainer.nnodes=$MLP_WORKER_NUM \
"

python3 -m verl.trainer.main_ppo \
    $CONFIG_ARGS \
    $DATA_ARGS \
    $ACTOR_ROLLOUT_REF_ARGS \
    $TRAINER_ARGS