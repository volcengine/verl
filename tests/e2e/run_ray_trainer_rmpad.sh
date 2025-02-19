#!/usr/bin/env bash

set -e -x

python3 tests/e2e/arithmetic_sequence/rl/main_trainer.py \
    data.train_files=tests/e2e/arithmetic_sequence/data/train.parquet \
    data.val_files=tests/e2e/arithmetic_sequence/data/test.parquet \
    actor_rollout_ref.model.path=tests/e2e/arithmetic_sequence/model \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.tokenizer_path=tests/e2e/arithmetic_sequence/model \
    critic.model.path=Qwen/Qwen2.5-0.5B \
    critic.model.use_remove_padding=True \
    trainer.total_epochs=1