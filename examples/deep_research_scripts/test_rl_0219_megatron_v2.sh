set -x

# export WANDB_ENTITY=glm-zero
# export WANDB_BASE_URL=https://wandb.glm.ai
# export WANDB_API_KEY=local-515145080efc923cf7bd1427ed76cda1c83a15c5
# export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_ENTITY=deep_research
export WANDB_BASE_URL=https://wandb.ai
export WANDB_API_KEY=06deee090a842fccccbdc8569567287f3725339b
export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_PROJECT=deep_research_rl
export EXP_NAME=$(basename "$0" .sh)

TRAIN_FILE=/workspace/lurui-yun/deep_research/prompts/res/hotpotQA/train.parquet
TEST_FILE=/workspace/lurui-yun/deep_research/prompts/res/hotpotQA/test.parquet
# MODEL_PATH=/data/o1-cloud/OpenSourceModels/Qwen2.5-7B-Instruct
MODEL_PATH=/data/o1-cloud/lurui/checkpoint/9b_simple_hf_epoch_1_0218

# GLM 模型 use_remove_padding=False

python3 -m verl.trainer.main_ppo \
    --config-path=./config --config-name='ppo_megatron_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    +trainer.val_before_train=False \
    trainer.total_epochs=2 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_NAME $@ \

# +model.trust_remote_code=True \