set -x

# micro_batch_size = train_batch_size * rollout * sp / 8 
# 8 * 2 * 2  / 8 = 4
# for personal
# export WANDB_API_KEY=8079db660b5be78f814fa9dab054e1a784185f67
export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_PROJECT=test_verl
export EXP_NAME=$(basename "$0" .sh)

DATASET_PREFIX=/workspace/haoran-cloud/verl-deep-research/datasets
TRAIN_FILE=$DATASET_PREFIX/debug.parquet
TEST_FILE=$DATASET_PREFIX/debug.parquet

MODEL_PATH=/workspace/haoran/models/Qwen2.5-Coder-7B-Instruct-SWE-0203-1e-5_bs64_3_32k
SAVE_PATH=/workspace/haoran-cloud/verl-deep-research/test

python3 -m verl.trainer.main_ppo \
    --config-path=/workspace/haoran-cloud/verl-deep-research/verl/trainer/config \
    --config-name='ppo_trainer_agent' \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=10000 \
    data.max_response_length=20000 \
    data.shuffle=False \
    +data.task_type=swedev \
    +data.is_swedev=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    +actor_rollout_ref.rollout.sampling_params.temperature=0.9 \
    +actor_rollout_ref.rollout.sampling_params.top_p=0.9 \
    +actor_rollout_ref.rollout.sampling_params.max_new_tokens=1024 \
    +actor_rollout_ref.rollout.task_type=swedev \
    actor_rollout_ref.rollout.n=2 \
    +actor_rollout_ref.rollout.is_swedev=True \
    actor_rollout_ref.rollout.multi_turn=True \
    actor_rollout_ref.rollout.prompt_length=26144 \
    actor_rollout_ref.rollout.response_length=5000 \
    actor_rollout_ref.rollout.max_turns=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=$MLP_GPU \
    trainer.nnodes=$MLP_WORKER_NUM \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=4 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.experiment_name=$EXP_NAME $@ \
    trainer.val_before_train=False \
    reward_model.reward_manager=swedev

# +actor_rollout_ref.model.trust_remote_code=True \
