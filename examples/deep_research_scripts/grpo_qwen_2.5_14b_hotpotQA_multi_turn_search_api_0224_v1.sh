set -x

allrun cp /data/o1-cloud/lurui/verl/registry.py /root/miniconda3/lib/python3.10/site-packages/vllm/inputs/registry.py

# for team
# export WANDB_ENTITY=glm-zero
# export WANDB_BASE_URL=https://wandb.glm.ai
# export WANDB_API_KEY=local-515145080efc923cf7bd1427ed76cda1c83a15c5
# export VLLM_ATTENTION_BACKEND=XFORMERS

# for personal
export WANDB_ENTITY=deep_research
export WANDB_BASE_URL=https://wandb.ai
export WANDB_API_KEY=06deee090a842fccccbdc8569567287f3725339b
export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_PROJECT=deep_research_rl
export EXP_NAME=$(basename "$0" .sh)

DATASET_PREFIX=/workspace/lurui-yun/deep_research/prompts/res/hotpotQA_qwen_system
TRAIN_FILE=$DATASET_PREFIX/train.parquet
TEST_FILE=$DATASET_PREFIX/test.parquet

# MODEL_PATH=/data/o1-cloud/lurui/checkpoint/9b_simple_hf_epoch_1_0218
MODEL_PATH=/data/o1-cloud/yujiang/openrlhf-qwencp/ckpt/qwen-glm-template/epoch_1
SAVE_PATH=/workspace/ckpt/lurui_verl/ckpt/$WANDB_PROJECT/$EXP_NAME

CONFIG_ARGS="
    --config-path=./config \
    --config-name="ppo_trainer_agent" \
"

ALGORITHM_ARGS="
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
"

# world_size | train_batch_size, val_batch_size
WORLD_SIZE=$((MLP_WORKER_NUM * MLP_GPU))
BATCH_SIZE=$WORLD_SIZE
DATA_ARGS="
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.shuffle=False \
"

# world_size | actor_mini_batch_size
NUM_TRACES=4
ACTOR_MICRO_BATCH_SIZE=$((BATCH_SIZE * NUM_TRACES / WORLD_SIZE))
TP=4

ACTOR_ROLLOUT_REF_ARGS="
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$WORLD_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.multi_turn=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=$NUM_TRACES \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
"

TRAINER_ARGS="
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=$MLP_GPU \
    trainer.nnodes=$MLP_WORKER_NUM \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=4 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.experiment_name=$EXP_NAME $@ \
    trainer.val_before_train=False \
"

python3 -m verl.trainer.main_ppo \
    $CONFIG_ARGS \
    $ALGORITHM_ARGS \
    $DATA_ARGS \
    $ACTOR_ROLLOUT_REF_ARGS \
    $TRAINER_ARGS