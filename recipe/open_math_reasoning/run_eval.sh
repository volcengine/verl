#!/usr/bin/env bash

# Evaluation
python3 -m verl.trainer.main_eval \
    data.path=$HOME/data/gen/qwen_8b_gen_test.parquet \
    custom_reward_function.path=recipe/open_math_reasoning/compute_score.py \
    custom_reward_function.name=compute_score_data_source
