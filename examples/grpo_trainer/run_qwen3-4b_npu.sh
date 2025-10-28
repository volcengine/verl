source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -xeuo pipefail

trainer_n_gpus_per_node=16
trainer_nnodes=1
log_path=logs/${trainer_project_name}/${trainer_experiment_name}.log
use_dynamic_bsz=True
project_name='GRPO-Qwen3'
exp_name='GRPO-Qwen3-4B-npu'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-4B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k.parquet"}

# 开启二级流水
export TASK_QUEUE_ENABLE=2
# 开启帮核
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_VERSION=0.9.1
export TENSORBOARD_DIR=tb/${trainer_project_name}/${trainer_experiment_name}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
	  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.logger=['console','tensorboard'] \
    trainer.default_local_dir=${CKPTS_DIR} \ \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    trainer.total_training_steps=5 \
    trainer.device=npu 2>&1 | tee $log_path