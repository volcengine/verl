#!/usr/bin/env bash

DATA_PATH=/opt/tiger/datasets/r1_bench
MODEL_PATH=/mnt/hdfs/zhangchi.usc1992_ssd_hldy/open_verl/sft/verl_sft_test/nvidia-openmathreasoning-fsdp-fsdp2-sp8-fsdp16-pad-no_padding-use_remove_padding-True/global_step_11783/huggingface

# Eval Data Process
python3 -m recipe.r1.data_process \
    --local_dir $DATA_PATH \
    --tasks aime2024,gpqa_diamond


# Generation
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=16 \
    data.path=$DATA_PATH/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/test-output-8.parquet \
    '+data.apply_chat_template_kwargs={enable_thinking:True}' \
    model.path=$MODEL_PATH \
    rollout.temperature=0.6 \
    rollout.top_p=0.95 \
    rollout.prompt_length=1024 \
    rollout.response_length=20480 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536