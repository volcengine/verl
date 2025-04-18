#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}

ENGINE=${ENGINE:-vllm}
RM_PAD=${RM_PAD:-True}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
USE_KL=${USE_KL:-False}
CUSTOM_REWARD_FN=${CUSTOM_REWARD_FN:-False}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True} # For vLLM VLM placeholder issue: https://github.com/vllm-project/vllm/issues/15185
# Validation
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
# Save & Resume
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:--1}
TOT_TRAIN_STEPS=${TOT_TRAIN_STEPS:-1}

train_traj_micro_bsz_per_gpu=2 # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

reward_fn_name=null
reward_fn_file_path=null
output_file="$(pwd)/output.txt"
if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    reward_fn_name="my_reward_function"
    reward_fn_file_path="$(pwd)/my_reward_function.py"
    rm -rf "${reward_fn_file_path}"
    cat <<EOF > "$reward_fn_file_path"
def ${reward_fn_name}(data_source, solution_str, ground_truth, extra_info=None):
    print(f"Congratulations!!! You have called ${reward_fn_name} successfully!!!")
    return 0.1
EOF

    rm -rf "${output_file}"
fi

exp_name="$(basename "${MODEL_ID,,}")-function-reward-minimal"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    custom_reward_function.path="${reward_fn_file_path}"\
    custom_reward_function.name="${reward_fn_name}"\
    algorithm.use_kl_in_reward="${USE_KL}" \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOT_TRAIN_STEPS}" $@ \
    | tee "${output_file}"

if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    python3 tests/e2e/check_custom_rwd_fn.py --output_file="${output_file}"
    check_exit_code=$?
    rm -rf "${reward_fn_file_path}"
    rm -rf "${output_file}"
    # Return the exit code of check_custom_rwd_fn.py if it fails
    if [ $check_exit_code -ne 0 ]; then
        exit $check_exit_code
    fi
fi