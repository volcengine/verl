#!/usr/bin/env bash
set -uxo pipefail

if [ -z "${DATA_HOME}" ]; then
  echo "DATA_HOME is not set"
  exit 1
fi

export TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME}/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${DATA_HOME}/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

# filter duplicate rows of original DAPO dataset
if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/pe-nlp/DAPO-Math-17k/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/pe-nlp/DAPO-AIME-2024/resolve/main/data/train-00000-of-00001.parquet?download=true"
fi