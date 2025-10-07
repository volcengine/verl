#!/bin/bash

MODEL_NAME=Qwen/Qwen2.5-7B

huggingface-cli download $MODEL_NAME \
  --repo-type model \
  --resume-download \
  --local-dir downloads/$MODEL_NAME \
  --local-dir-use-symlinks False \
  --exclude *.pth