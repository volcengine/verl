#!/usr/bin/env bash

set -e -x

OUTPUT_FILE="/tmp/output_ray_trainer.txt"

export PATH=$PATH:~/.local/bin

rm -rf $OUTPUT_FILE
python3 tests/e2e/arithmetic_sequence/rl/main_trainer.py \
    data.train_files=tests/e2e/arithmetic_sequence/data/train.parquet \
    data.val_files=tests/e2e/arithmetic_sequence/data/test.parquet \
    actor_rollout_ref.model.path=tests/e2e/arithmetic_sequence/model \
    critic.model.path=tests/e2e/arithmetic_sequence/model | tee $OUTPUT_FILE;

python3 tests/e2e/check_results.py --output_file=$OUTPUT_FILE
rm -rf $OUTPUT_FILE
