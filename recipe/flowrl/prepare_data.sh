#!/usr/bin/env bash
set -uxo pipefail

export DATA_DIR=${DATA_DIR:-"downloads"}

# Download DAPO-Math-17k dataset
DATASET_NAME_TRAIN="BytedTsinghua-SIA/DAPO-Math-17k"
huggingface-cli download $DATASET_NAME_TRAIN \
  --repo-type dataset \
  --resume-download \
  --local-dir ${DATA_DIR}/${DATASET_NAME_TRAIN} \
  --local-dir-use-symlinks False

# Download AIME-2024 dataset
DATASET_NAME_TEST="BytedTsinghua-SIA/AIME-2024"
huggingface-cli download $DATASET_NAME_TEST \
  --repo-type dataset \
  --resume-download \
  --local-dir ${DATA_DIR}/${DATASET_NAME_TEST} \
  --local-dir-use-symlinks False
