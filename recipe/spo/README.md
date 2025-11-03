# Single-stream Policy Optimization (SPO)

[![arXiv](https://img.shields.io/badge/arXiv-2509.13232-b31b1b.svg)](https://arxiv.org/abs/2509.13232)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

### Prerequisites

- Python 3.12
- CUDA 12.8 compatible GPU
- Conda or Mamba package manager

### Setup Instructions

1. **Clone the VERL repository at the specific commit:**

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout d7944c01e63e9eb639c8357648b7958550591158
```

2. **Create and activate a new conda environment:**

```bash
conda create -n spo python=3.12 -y
conda activate spo
```

3. **Install dependencies:**

```bash
# Install vLLM with CUDA 12.8 support
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention
pip install --no-cache-dir --no-build-isolation flash_attn==2.7.4.post1

# Install verl
pip install -e .
```

### Environment Reference

For a complete list of dependencies and package versions, see [`environment.yml`](./environment.yml). This file contains the full conda environment export and can be used as a reference for troubleshooting dependency issues.

## Offline Value Estimation

Offline value estimation is a crucial preprocessing step in SPO that estimates the quality of responses in your training dataset using a pretrained model. This process helps initialize the value function for more efficient policy optimization.

### Step 1: Preprocess Training Data

First, split your training dataset into manageable subsets using the preprocessing script:

```bash
python recipe/spo/estimate_offline_values/split_dapo_into_subsets.py \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --output_dir DAPO-Math-17k-Processed_Splits \
    --num_subsets 5
```

**Parameters:**
- `--dataset`: HuggingFace dataset identifier or local path (default: `open-r1/DAPO-Math-17k-Processed`)
- `--output_dir`: Directory where subset parquet files will be saved (required)
- `--num_subsets`: Number of subsets to split the dataset into (default: 5)

This script will generate multiple subset `.parquet` files under the specified `output_dir`. For example:
- `DAPO-Math-17k-Processed_Splits/subset_0.parquet`
- `DAPO-Math-17k-Processed_Splits/subset_1.parquet`
- `DAPO-Math-17k-Processed_Splits/subset_2.parquet`
- `DAPO-Math-17k-Processed_Splits/subset_3.parquet`
- `DAPO-Math-17k-Processed_Splits/subset_4.parquet`

### Step 2: Generate Offline Value Estimates

Run the evaluation script to generate offline value estimates using a pretrained model. You'll need to process each subset individually:

```bash
OUTPUT_DIR=spo_verl_pr \
DATA_FILE=DAPO-Math-17k-Processed_Splits/subset_0.parquet \
MODEL_PATH=Qwen/Qwen3-8B \
EXP_NAME=offline_value_estimation_subset_0 \
sh recipe/spo/estimate_offline_values/eval.sh
```

**Parameters:**
- `OUTPUT_DIR`: Directory where results will be saved
- `DATA_FILE`: Path to the subset parquet file to process
- `MODEL_PATH`: HuggingFace model identifier or local path to the pretrained model
- `EXP_NAME`: Experiment name for tracking and organizing results

**Batch Processing:**

To process all subsets, you can loop through them:

```bash
for i in {0..N}; do
    OUTPUT_DIR=spo_verl_pr \
    DATA_FILE=DAPO-Math-17k-Processed_Splits/subset_${i}.parquet \
    MODEL_PATH=Qwen/Qwen3-8B \
    EXP_NAME=offline_value_estimation_subset_${i} \
    sh recipe/spo/estimate_offline_values/eval.sh
done
```

Replace `N` with the actual number of subsets generated in Step 1.
