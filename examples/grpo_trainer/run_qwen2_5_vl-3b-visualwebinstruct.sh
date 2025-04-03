#!/bin/bash
set -x

# 定义环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=""
export WANDB_PROJECT="vlm_visualwebinstruct_rlhf"
export HDFS_CHECKPOINT_PATH="/data/yiming/data/RL_checkpoints"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 可配置参数
ENGINE=${1:-vllm}
RUN_NAME="qwen2_5_vl_7b_visualwebinstruct_verifier"
VLM_MODEL_PATH="/data/yiming/data/qwen2_5vl_3b"
VERIFIER_MODEL_PATH="/data/yiming/data/verifier-full-qwen2.5-math-1.5b"  
DATA_DIR="/data/yiming/data/visualwebinstruct_verified/RL_1image"
# DATA_DIR="/data/yiming/data/geometry3k"
NUM_GPUS=6
NUM_NODES=1

# 运行训练脚本
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.enable=True \
    reward_model.model.input_tokenizer=$VERIFIER_MODEL_PATH \
    reward_model.model.path=$VERIFIER_MODEL_PATH \
    reward_model.strategy=verifier \
    reward_model.reward_manager=naive \
    reward_model.micro_batch_size=96 \
    +reward_model.n_gpus_per_node_for_rm=2 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.prompt_key=prompt \
    data.max_prompt_length=8192 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.truncation='left' \
    data.image_key=images \
    actor_rollout_ref.model.path=$VLM_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_ctrl.type=adaptive \
    +algorithm.kl_ctrl.horizon=10000 \
    +algorithm.kl_ctrl.target_kl=0.01 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=32 \
    critic.optim.lr=5e-6 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vlm_visualwebinstruct_grpo' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.val_generations_to_log_to_wandb=10 \
    trainer.total_epochs=2 $@