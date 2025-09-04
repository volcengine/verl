#!/usr/bin/env bash
set -uxo pipefail

cur_path=$PWD

export TRAIN_FILE=${TRAIN_FILE:-"${cur_path}/data/deepscaler_preview.parquet"}
export TEST_FILE=${TEST_FILE:-"${cur_path}/data/aime24.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${cur_path}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/ganglii/DeepScaleR-Preview-Dataset/resolve/main/deepscaler_preview.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/ganglii/AIME24/resolve/main/aime24.parquet?download=true"
fi
