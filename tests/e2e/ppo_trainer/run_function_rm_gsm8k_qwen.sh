#!/usr/bin/env bash
set -x

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-$HOME/models/${MODEL_ID}}

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}

ENGINE=${ENGINE:-vllm}
RM_PAD=${RM_PAD:-True}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
USE_KL=${USE_KL:-False}
CUSTOM_REWARD_FN=${CUSTOM_REWARD_FN:-False}

reward_fn_name=null
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

huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding="${RM_PAD}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding="${RM_PAD}" \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    custom_reward_function.path="${reward_fn_file_path}"\
    custom_reward_function.name="${reward_fn_name}"\
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name='qwen_e2e_ci_function_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=2 $@ \
    | tee "${output_file}";

if [ "${CUSTOM_REWARD_FN}" = "True" ]; then
    python3 tests/e2e/check_custom_rwd_fn.py --output_file="${output_file}"
    rm -rf "${reward_fn_file_path}"
    rm -rf "${output_file}"
fi