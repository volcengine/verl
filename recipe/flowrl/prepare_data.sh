#!/usr/bin/env bash
set -uxo pipefail

export DATA_DIR=${DATA_DIR:-"downloads"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${DATA_DIR}/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${DATA_DIR}"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi
