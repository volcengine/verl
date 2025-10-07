#!/usr/bin/env bash
set -uxo pipefail

export DOWNLOAD_DIR=${DOWNLOAD_DIR:-"downloads"}
export DATA_DIR=${DATA_DIR:-"${DOWNLOAD_DIR}/data"}

# Create final data directory
mkdir -p "${DATA_DIR}"

# Download DAPO-Math-17k dataset
DATASET_NAME_TRAIN="BytedTsinghua-SIA/DAPO-Math-17k"
echo "Downloading ${DATASET_NAME_TRAIN}..."
huggingface-cli download $DATASET_NAME_TRAIN \
  --repo-type dataset \
  --resume-download \
  --local-dir ${DOWNLOAD_DIR}/${DATASET_NAME_TRAIN} \
  --local-dir-use-symlinks False

# Move the parquet file to data directory
if [ -f "${DOWNLOAD_DIR}/${DATASET_NAME_TRAIN}/data/dapo-math-17k.parquet" ]; then
  mv "${DOWNLOAD_DIR}/${DATASET_NAME_TRAIN}/data/dapo-math-17k.parquet" "${DATA_DIR}/dapo-math-17k.parquet"
  echo "✓ Moved dapo-math-17k.parquet to ${DATA_DIR}/"
fi

# Download AIME-2024 dataset
DATASET_NAME_TEST="BytedTsinghua-SIA/AIME-2024"
echo "Downloading ${DATASET_NAME_TEST}..."
huggingface-cli download $DATASET_NAME_TEST \
  --repo-type dataset \
  --resume-download \
  --local-dir ${DOWNLOAD_DIR}/${DATASET_NAME_TEST} \
  --local-dir-use-symlinks False

# Move the parquet file to data directory
if [ -f "${DOWNLOAD_DIR}/${DATASET_NAME_TEST}/data/aime-2024.parquet" ]; then
  mv "${DOWNLOAD_DIR}/${DATASET_NAME_TEST}/data/aime-2024.parquet" "${DATA_DIR}/aime-2024.parquet"
  echo "✓ Moved aime-2024.parquet to ${DATA_DIR}/"
fi

echo ""
echo "Data preparation completed!"
echo "Training file: ${DATA_DIR}/dapo-math-17k.parquet"
echo "Test file: ${DATA_DIR}/aime-2024.parquet"
