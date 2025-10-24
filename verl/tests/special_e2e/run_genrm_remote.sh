#!/usr/bin/env bash

export no_proxy="localhost,127.0.0.1"

set -x

# Launch a vllm server
CUDA_VISIBLE_DEVICES=0 vllm serve $HOME/models/verl-team/GenRM-CI-Test-1.5B \
    --served_model_name genrm-demo --host localhost --port 30000 > /dev/null &
SERVER_PID=$!

# kill server when script exits
cleanup() {
    echo "Cleaning up..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Cleanup done"
}
trap cleanup EXIT

# wait for server to start
wait_for_server() {
    local max_attempts=60
    local attempt=0
    local sleep_time=10

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:30000/health" >/dev/null; then
            echo "Server is up and running!"
            return 0
        fi
        echo "Waiting for server to start... (attempt $((attempt+1))/$max_attempts)"
        sleep $sleep_time
        ((attempt++))
    done
    
    echo "Error: Failed to start server after $max_attempts attempts" >&2
    return 1
}

if ! wait_for_server; then
    exit 1
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${HOME}/data/gsm8k/train.parquet \
    data.val_files=${HOME}/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HOME/models/Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=recipe/genrm_remote/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl-test' \
    trainer.experiment_name='qwen2.5-0.5b-gen-rm' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=10 \
    trainer.resume_mode='disable' \
    trainer.total_training_steps=1
