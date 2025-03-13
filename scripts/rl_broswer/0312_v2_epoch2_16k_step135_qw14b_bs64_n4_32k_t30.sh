set -x

# for debug
export HYDRA_FULL_ERROR=1

export WANDB_API_KEY=218cc9a2633c3b2303ca4dbc44397b10fa3e9115
export RAY_DEDUP_LOGS_ALLOW_REGEX="nodedup"
export VERL_PPO_LOGGING_LEVEL="INFO"

WANDB_PROJECT=research_rl
EXP_NAME=$(basename "$0" .sh)

MODEL_PATH=/workspace/ckpt/lurui_verl/ckpt/research_rl/0312_v1_epoch2_qw14b_bs64_n4_16k_t30/global_step_135/actor/huggingface_to_train
SAVE_PATH=/workspace/ckpt/lurui_verl/ckpt/$WANDB_PROJECT/$EXP_NAME

CONFIG_ARGS="
    --config-path=$(pwd)/configs \
    --config-name=qwen14b_sft_async \
"

# WARNING: skip 9k data!
DATASET_PREFIX=/workspace/lurui-yun/deep_research/prompts/res/hotpotQA_grok_system_harder_yst_14k_0313
WORLD_SIZE=$((MLP_WORKER_NUM * MLP_GPU))
BATCH_SIZE=$WORLD_SIZE

CONTEXT_LENGTH=32768

DATA_ARGS="
    data.train_files=$DATASET_PREFIX/train.parquet \
    data.val_files=$DATASET_PREFIX/test.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.max_response_length=$CONTEXT_LENGTH \
"

ACTOR_ROLLOUT_REF_ARGS="
    actor_rollout_ref.actor.ppo_mini_batch_size=$WORLD_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
"

TRAINER_ARGS="
    trainer.n_gpus_per_node=$MLP_GPU \
    trainer.nnodes=$MLP_WORKER_NUM \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$SAVE_PATH \
"

python3 -m verl.trainer.main_ppo \
    $CONFIG_ARGS \
    $DATA_ARGS \
    $ACTOR_ROLLOUT_REF_ARGS \
    $TRAINER_ARGS