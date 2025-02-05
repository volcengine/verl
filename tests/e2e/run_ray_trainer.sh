#!/usr/bin/env bash

set -e -x

OUTPUT_FILE="/tmp/output_ray_trainer.txt"

export PATH=$PATH:~/.local/bin

rm -rf $OUTPUT_FILE
python3 tests/e2e/arithmetic_sequence/rl/main_trainer.py \
    data.train_files=tests/e2e/arithmetic_sequence/data/train.parquet \
    data.val_files=tests/e2e/arithmetic_sequence/data/test.parquet \
    data.train_batch_size=800 \
    data.val_batch_size=200 \
    data.max_prompt_length=16 \
    data.max_response_length=32 \
    data.return_raw_input_ids=True \
    actor_rollout_ref.model.path=tests/e2e/arithmetic_sequence/model \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=200 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=200 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=200 \
    actor_rollout_ref.rollout.name=hf \
    critic.ppo_micro_batch_size_per_gpu=200 \
    critic.model.path=tests/e2e/arithmetic_sequence/model | tee $OUTPUT_FILE;

python3 tests/e2e/check_results.py --output_file=$OUTPUT_FILE
rm -rf $OUTPUT_FILE
