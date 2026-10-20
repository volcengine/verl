#!/usr/bin/env bash
# Optimal Token Baseline | Qwen2.5-7B base | vLLM rollout | Megatron training | Ascend NPU
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=false
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export RAY_DEDUP_LOGS=0
export ASCEND_GLOBAL_LOG_LEVEL=3
unset ASCEND_MODULE_LOG_LEVEL
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# CANN Python modules must also be available to rollout worker processes.
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"

# Ray assigns each worker a single local device rank.
case "${LOCAL_RANK:-}" in
    ""|*[!0-9]*) unset LOCAL_RANK ;;
esac

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-optimal_token_baseline}
DATA_ROOT=${DATA_ROOT:-${HOME}/data}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}/outputs/qwen2_5_7b_otb}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

train_files="['${DATA_ROOT}/gsm8k/train.parquet','${DATA_ROOT}/math/train.parquet']"
val_files="['${DATA_ROOT}/gsm8k/test.parquet','${DATA_ROOT}/math/test.parquet']"

hydra_args=()
while (($#)); do
    case "$1" in
        --checkpoint_url|--data_url)
            shift
            if (($#)); then
                shift
            fi
            ;;
        --checkpoint_url=*|--data_url=*)
            shift
            ;;
        *)
            hydra_args+=("$1")
            shift
            ;;
    esac
done

USER_LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}
RUN_STARTED_AT=$(date '+%Y%m%d-%H%M%S')
RUN_LOG_DIR="${USER_LOG_DIR}/otb_${RUN_STARTED_AT}_pid$$"
RUN_LOG_FILE="${RUN_LOG_DIR}/trainer.log"
METRICS_LOG_FILE="${RUN_LOG_DIR}/metrics.log"
export VERL_FILE_LOGGER_PATH="${METRICS_LOG_FILE}"

mkdir -p "${RUN_LOG_DIR}"
printf 'Training log directory: %s\n' "${RUN_LOG_DIR}" | tee -a "${RUN_LOG_FILE}"
printf 'Training stdout/stderr log: %s\n' "${RUN_LOG_FILE}" | tee -a "${RUN_LOG_FILE}"
printf 'Training metrics JSONL log: %s\n' "${METRICS_LOG_FILE}" | tee -a "${RUN_LOG_FILE}"

set +e
python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.calculate_sum_pi_squared=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.ascend_compilation_config.enable_npugraph_ex=False \
    trainer.device=npu \
    trainer.use_v1=False \
    trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name=verl_grpo_example_gsm8k \
    trainer.experiment_name=qwen2_5-7b-otb \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
    trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=100 \
    "${hydra_args[@]}" 2>&1 | tee -a "${RUN_LOG_FILE}"
training_status=${PIPESTATUS[0]}
printf 'Training process exit code: %s\n' "${training_status}" | tee -a "${RUN_LOG_FILE}"
set -e
exit "${training_status}"
