# Recipe for InfiGUI-G1

This directory contains the official implementation for the paper [InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](https://arxiv.org/abs/2508.05731).

This work introduces Adaptive Exploration Policy Optimization (AEPO), a policy optimization framework designed to enhance GUI grounding in Multimodal Large Language Models (MLLMs). AEPO improves exploration efficiency by employing a multi-answer generation strategy and a theoretically grounded Adaptive Exploration Reward (AER) function. This approach effectively addresses the challenge of semantic alignment in complex GUI grounding tasks.

We provide training scripts for both 3B and 7B models, configured for a single machine with 8 GPUs by default.

## Environment Setup

Please follow the main environment setup guide for `verl`.

The provided scripts use the following Docker image: `verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2`

## Data Preparation

Before starting the training, you need to download the example dataset. This dataset is a filtered version of [omniact](https://huggingface.co/datasets/Writer/omniact), containing only grounding tasks and excluding easy samples.

The data is hosted on the Hugging Face. You can download it using the `huggingface-cli`:

```bash
huggingface-cli download --repo-type dataset --resume-download InfiX-ai/omniact_grounding_filtered --local-dir data/omniact_grounding_filtered
```

This command will download the training and validation parquet files into the `data/omniact_grounding_filtered` directory, which is the default path used by the scripts.

## Training

We provide scripts to train the 3B and 7B models. Please run them from the root directory of `verl`.

-   **Train the 3B model:**

    ```bash
    bash recipe/infigui-g1/run_3b.sh
    ```

-   **Train the 7B model:**

    ```bash
    bash recipe/infigui-g1/run_7b.sh
    ```

## Using Custom Data

If you wish to train on your own dataset, please format your data to match the structure of the example files located in `data/omniact_grounding_filtered`.

Once your data is ready, you need to update the data path arguments in the training script.

In `run_3b.sh` or `run_7b.sh`, modify the following lines:

```bash
    data.train_files=./path/to/your/train_data.parquet \
    data.val_files=./path/to/your/val_data.parquet \
```

Replace the paths with the location of your custom data files.
