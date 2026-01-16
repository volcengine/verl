set -x

DATE_TIME=$(date +%Y%m%d_%H%M%S)

project_name='verl_exp_partial_rollout_dapo_math'
exp_name="v070-qwen3-4b-sample10k-nopr-grpo-bs64-${DATE_TIME}"

# Paths
RAY_DATA_HOME=/apdcephfs_gy2/share_303055091/allenzpma_tmp
MODEL_PATH="/apdcephfs_gy2/share_303055091/Qwen3-4B"
CKPTS_DIR="${RAY_DATA_HOME}/checkpoint/${project_name}/${exp_name}"
TRAIN_FILE="${RAY_DATA_HOME}/data/dapo-math17k/DAPO-Math-17k_25k.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/aime/DAPO-Math-AIME-2024.parquet"
LOG_PATH="${RAY_DATA_HOME}/partial_rollout/output/logs"

NNODES=1
NGPUS_PER_NODE=8

export RAY_DEBUG=1
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export SWANLAB_API_KEY=7YvciQHdp4AlO8bF0lHVu

# For async rollout mode, dataset should return raw chat.
rollout_mode="async" # sync or async，async会单独判断
rollout_name="vllm" # sglang or vllm

if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=2048
max_response_length=8192

train_prompt_bsz=128
train_prompt_mini_bsz=64

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=2
sp_size=2
fsdp_size=4


python3 -m verl.trainer.main_ppo \
    --config-path=config \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat="${return_raw_chat}" \
    rollout.partial_rollout.enable=False \
    rollout.partial_rollout.max_chunk_response_length=${max_response_length} \
    rollout.partial_rollout.clip_tis_c=null \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.name="${rollout_name}" \
    actor_rollout_ref.rollout.mode="${rollout_mode}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${CKPTS_DIR}" \
    trainer.total_epochs=1 $@ 2>&1 | tee -a ${LOG_PATH}/log_${DATE_TIME}.txt