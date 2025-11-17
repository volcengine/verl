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

### Sandbox Runtime

For instructions on setting up and serving the Sandbox runtime environment, see the [verl-reTool recipe documentation](https://www.notion.so/verl-reTool-recipe-2398b5b7feba80a58156fa936f9f8de6).

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

**Output Directory Structure**

All subset outputs are saved in the `trainer.validation_data_dir` directory. The directory structure will look like this:

```
offline_value_estimation/
├── offline_value_estimation_subset_0
│   └── validation_data
│       └── 0.jsonl
├── offline_value_estimation_subset_1
│   └── validation_data
│       └── 0.jsonl
├── offline_value_estimation_subset_2
│   └── validation_data
│       └── 0.jsonl
├── offline_value_estimation_subset_3
│   └── validation_data
│       └── 0.jsonl
└── offline_value_estimation_subset_4
    └── validation_data
        └── 0.jsonl
```

Each subset directory contains:
- A `validation_data` subdirectory with the estimated values stored in JSONL format
- The `0.jsonl` file contains the offline value estimates for each response in the corresponding subset

### Step 3: Merge Offline Value Estimates

After generating offline value estimates for all subsets, merge them into a single file for downstream training:

```bash
python recipe/spo/estimate_offline_values/merge_offline_values.py \
    --input_dir offline_value_estimation \
    --output_file offline_values.json
```

**Parameters:**
- `--input_dir`: Directory containing all subset outputs (the `trainer.validation_data_dir` from Step 2)
- `--output_file`: Path where the merged offline values JSON file will be saved
- `--pattern`: (Optional) Custom glob pattern to match subset result files (default: `offline_value_estimation_subset_*/validation_data/0.jsonl`)
- `--max_scores_per_prompt`: (Optional) Maximum number of scores to keep per prompt. If a prompt has more scores, they will be randomly subsampled (default: 8)

**Output Format:**

The merged file contains a dictionary mapping prompts to their corresponding offline value scores:

```json
{
  "prompt_1": [score_1, score_2, ...],
  "prompt_2": [score_1, score_2, ...],
  ...
}
```

**Example:**

```bash
# Merge all subsets from the default output directory
python recipe/spo/estimate_offline_values/merge_offline_values.py \
    --input_dir spo_verl_pr/offline_value_estimation \
    --output_file DAPO-Math-17k-Processed_Splits/offline_values.json

# With custom pattern
python recipe/spo/estimate_offline_values/merge_offline_values.py \
    --input_dir /path/to/validation_data_dir \
    --output_file /path/to/output/offline_values.json \
    --pattern "custom_subset_*/validation_data/0.jsonl"

# With custom max scores per prompt
python recipe/spo/estimate_offline_values/merge_offline_values.py \
    --input_dir spo_verl_pr/offline_value_estimation \
    --output_file DAPO-Math-17k-Processed_Splits/offline_values.json \
    --max_scores_per_prompt 16
```

The script will:
- Automatically discover all subset result files matching the pattern
- Use concurrent processing to efficiently load data from multiple files
- Merge scores by prompt/question
- Display statistics about the merged data
- Save the final merged results to the specified output file

## Training

SPO provides two training methods: **GRPO** (Group Relative Policy Optimization) and **SPO** (Single-stream Policy Optimization). Both methods use the same training script but with different configurations.

### Prerequisites

Before training, ensure you have:
1. Preprocessed training data split into subsets (from [Step 1](#step-1-preprocess-training-data))
2. For SPO method: Merged offline value estimates (from [Step 3](#step-3-merge-offline-value-estimates))

### Training with GRPO

GRPO is a group-based policy optimization method that generates multiple responses per prompt during training. This is the simpler baseline method that doesn't require offline value estimation.

```bash
OUTPUT_DIR=spo_verl_pr \
TRAIN_DATA_DIR=DAPO-Math-17k-Processed_Splits \
MODEL_PATH=Qwen/Qwen3-8B \
EXP_NAME=grpo_training \
METHOD=GRPO \
sh recipe/spo/train.sh
```

**GRPO Configuration:**
- Generates **8 responses** per prompt during training
- Training batch size: 96
- PPO mini-batch size: 12
- Generation batch size: 96 (matches training batch size)

### Training with SPO

SPO is the single-stream policy optimization method that uses offline value estimates for more efficient training. This method generates only one response per prompt and uses Thompson Sampling to select prompts based on their offline value estimates.

```bash
OUTPUT_DIR=spo_verl_pr \
TRAIN_DATA_DIR=DAPO-Math-17k-Processed_Splits \
MODEL_PATH=Qwen/Qwen3-8B \
EXP_NAME=spo_training \
METHOD=SPO \
OFFLINE_VALUES=DAPO-Math-17k-Processed_Splits/offline_values.json \
sh recipe/spo/train.sh
```

**SPO Configuration:**
- Generates **1 response** per prompt during training
- Training batch size: 768 (8x larger than GRPO)
- PPO mini-batch size: 96 (8x larger than GRPO)
- Generation batch size: 14,000 (for efficient batched generation)
- Requires offline values JSON file from preprocessing

### Training Parameters

All parameters are configured via environment variables:

**Required Parameters:**
- `OUTPUT_DIR`: Directory where results, checkpoints, and logs will be saved
- `TRAIN_DATA_DIR`: Directory containing training data subset parquet files (subset_0.parquet through subset_4.parquet)
- `MODEL_PATH`: HuggingFace model identifier or local path to the pretrained model
- `EXP_NAME`: Experiment name for tracking and organizing results
- `METHOD`: Training method, either `GRPO` or `SPO`

**SPO-Specific Parameters:**
- `OFFLINE_VALUES`: Path to the merged offline values JSON file (required when METHOD=SPO)

**Optional Parameters:**
- `RESPONSE_LENGTH`: Maximum response length in tokens (default: 8192)
- `N_TRAIN`: Number of responses per prompt for training with GRPO (default: 8, overridden to 1 for SPO)
- `N_VAL`: Number of responses per prompt for validation (default: 16)
- `DEBUG`: Enable debug mode with smaller batch sizes (default: False)
- `VAL_BEFORE_TRAIN`: Run validation before starting training (default: False)

### Output Directory Structure

Training outputs are organized in the following structure:

```
<OUTPUT_DIR>/
└── spo/
    └── <EXP_NAME>/
        ├── checkpoints/
        │   ├── epoch_0/
        │   ├── epoch_20/
        │   ├── epoch_40/
        │   └── ...
        ├── validation_data/
        │   ├── 0.jsonl
        │   ├── 10.jsonl
        │   ├── 20.jsonl
        │   └── ...
        └── tensorboard/
            └── events.out.tfevents.*
```

**Directory Contents:**
- `checkpoints/`: Model checkpoints saved every 20 epochs
- `validation_data/`: Validation results in JSONL format, saved every 10 epochs
- `tensorboard/`: TensorBoard logs for monitoring training progress

### Monitoring Training

View training progress in real-time using TensorBoard:

```bash
tensorboard --logdir <OUTPUT_DIR>/spo/<EXP_NAME>/tensorboard
```

Key metrics to monitor:
- `reward/mean`: Average reward across training samples
- `actor/loss`: Actor model loss
- `actor/lr`: Learning rate
- `validation/accuracy`: Validation accuracy on AIME 2024 and 2025 datasets

### Example: Complete SPO Training Pipeline

Here's a complete example combining all preprocessing and training steps:

```bash
# Step 1: Split dataset into subsets
python recipe/spo/estimate_offline_values/split_dapo_into_subsets.py \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --output_dir DAPO-Math-17k-Processed_Splits \
    --num_subsets 5

# Step 2: Generate offline value estimates for each subset
for i in {0..4}; do
    OUTPUT_DIR=spo_verl_pr \
    DATA_FILE=DAPO-Math-17k-Processed_Splits/subset_${i}.parquet \
    MODEL_PATH=Qwen/Qwen3-8B \
    EXP_NAME=offline_value_estimation_subset_${i} \
    sh recipe/spo/estimate_offline_values/eval.sh
done

# Step 3: Merge offline value estimates
python recipe/spo/estimate_offline_values/merge_offline_values.py \
    --input_dir spo_verl_pr/offline_value_estimation \
    --output_file DAPO-Math-17k-Processed_Splits/offline_values.json

# Step 4: Train with SPO
OUTPUT_DIR=spo_verl_pr \
TRAIN_DATA_DIR=DAPO-Math-17k-Processed_Splits \
MODEL_PATH=Qwen/Qwen3-8B \
EXP_NAME=spo_training \
METHOD=SPO \
OFFLINE_VALUES=DAPO-Math-17k-Processed_Splits/offline_values.json \
sh recipe/spo/train.sh
```