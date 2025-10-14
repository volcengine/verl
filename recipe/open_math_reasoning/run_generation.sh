#!/usr/bin/env bash

MODEL_PATH=/mnt/hdfs/zhangchi.usc1992_ssd_hldy/open_verl/sft/verl_sft_test/nvidia-openmathreasoning-qwen3-8b-fsdp-fsdp2-sp8-fsdp-1008a1/global_step_19751/huggingface

NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
OUTPUT_PATH=${OUTPUT_PATH:-$HOME/data/gen/qwen_8b_gen_test.parquet}
GEN_TP=${GEN_TP:-2}  # Default tensor parallel size to 2

aime24_test_path=${HOME}/data/math-ai/aime24_test.parquet
aime25_test_path=${HOME}/data/math-ai/aime25_test.parquet
train_files="['$aime24_test_path', '$aime25_test_path']"

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    data.train_files="$train_files" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \



