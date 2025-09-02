#!/bin/bash
# Example script for running multi-turn GRPO training on math problems

set -x

# Data paths - update these to your actual data locations
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# Run multi-turn GRPO training
python3 -m verl.trainer.main_ppo \
    --config-path=verl/trainer/config \
    --config-name=multiturn_grpo_math \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    algorithm.adv_estimator=grpo_multiturn \
    algorithm.multiturn_grpo.n_trajectories_per_question=6 \
    algorithm.multiturn_grpo.max_turns_per_trajectory=7 \
    multiturn_trajectory.context_condenser.keep_last_n_steps=3 \
    reward_computation.domain=math \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-7B-Instruct \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.temperature=0.3 \
    actor_rollout_ref.rollout.top_p=0.95 \
    trainer.project_name=verl_multiturn_grpo_math \
    trainer.experiment_name=qwen2_5_math_7b_multiturn_$(date +%Y%m%d_%H%M%S) \
    trainer.total_epochs=20 \
    trainer.test_freq=2 \
    trainer.save_freq=5 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    $@