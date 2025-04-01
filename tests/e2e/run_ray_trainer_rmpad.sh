#!/usr/bin/env bash

set -e -x

huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir $HOME/models/Qwen/Qwen2.5-0.5B

python3 tests/e2e/arithmetic_sequence/rl/main_trainer.py \
    algorithm.adv_estimator=gae \
    data.train_files=tests/e2e/arithmetic_sequence/data/train.parquet \
    data.val_files=tests/e2e/arithmetic_sequence/data/test.parquet \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.path=tests/e2e/arithmetic_sequence/model \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.tokenizer_path=tests/e2e/arithmetic_sequence/model \
    critic.model.path=Qwen/Qwen2.5-0.5B \
    critic.model.use_remove_padding=True \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=1