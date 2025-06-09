#!/bin/bash

export TORCHDYNAMO_DISABLE=1
export DISABLE_TRITON=1


MODEL_PATH="/home/yangkai/models/DeepSeek-R1-Distill-Qwen-7B"
PORT=8000

CUDA_VISIBLE_DEVICES=5 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $PORT \
    --dtype auto \
    --tokenizer $MODEL_PATH \
    --max-model-len 12000 