"""
Preprocess the DeepEyes dataset.

We should add some extra_info to use verl's multi-turn function calling.
"""

import argparse
import os

import pandas as pd
import datasets
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="path/to/local/dir")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    data_source = "hiyouga/DeepEyes-Datasets-47k"
    
    dataset = load_dataset(
        path=args.dataset_dir,
        data_files=["data_0.1.2_visual_toolbox_v2.parquet", "data_thinklite_reasoning_acc.parquet"],
    )

    def process_fn(example, idx):
        extra_info = example.pop("extra_info")
        extra_info["need_tools_kwargs"] = True
        extra_info["tools_kwargs"] = {
            "image_zoom_in_tool": {
                "create_kwargs": {"image": example["images"][0]},
            },
        }
        example["extra_info"] = extra_info
        return example

    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=8)
    
    # Split dataset: 1k for validation, rest for training
    train_test_split = dataset["train"].train_test_split(test_size=1000, seed=42)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]
    
    # Save train and validation datasets
    train_dataset.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.save_dir, "val.parquet"))
