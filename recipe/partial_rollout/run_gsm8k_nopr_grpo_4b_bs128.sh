set -x

DATE_TIME=$(date +%Y%m%d_%H%M%S)

project_name='verl_exp_partial_rollout_gsm8k'
exp_name="v070-qwen3-4b-gsm8k-nopr-grpo-bs128-${DATE_TIME}"

# Paths
RAY_DATA_HOME=/apdcephfs_gy2/share_303055091/allenzpma_tmp
MODEL_PATH="/apdcephfs_gy2/share_303055091/Qwen3-4B"
CKPTS_DIR="${RAY_DATA_HOME}/checkpoint/${project_name}/${exp_name}"
TRAIN_FILE="${RAY_DATA_HOME}/data/gsm8k/train.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/gsm8k/test.parquet"
LOG_PATH="${RAY_DATA_HOME}/partial_rollout/output/logs"

NNODES=1
NGPUS_PER_NODE=8

export RAY_DEBUG=1
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export SWANLAB_API_KEY=7a3mP8NZpdryp1nldFtWd

# For async rollout mode, dataset should return raw chat.
rollout_mode="async" # sync or async，async会单独判断
rollout_name="vllm" # sglang or vllm

if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${rollout_name}" \
    actor_rollout_ref.rollout.mode="${rollout_mode}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${CKPTS_DIR}" \
    trainer.total_epochs=5 $@ 2>&1 | tee -a ${LOG_PATH}/log_${DATE_TIME}.txt