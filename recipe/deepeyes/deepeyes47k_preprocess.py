"""
Preprocess the DeepEyes dataset.

We should add some extra_info to use verl's multi-turn function calling.
"""

import argparse
import os

import pandas as pd
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="path/to/local/dir")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    data_source = "hiyouga/DeepEyes-Datasets-47k"
    
    vstar_dataset = pd.read_parquet(os.path.join(args.dataset_dir, "data_0.1.2_visual_toolbox_v2.parquet"))
    chart_dataset = pd.read_parquet(os.path.join(args.dataset_dir, "data_v0.8_visual_toolbox_v2.parquet"))
    thinklite_dataset = pd.read_parquet(os.path.join(args.dataset_dir, "data_thinklite_reasoning_acc.parquet"))
    chart_dataset.drop(columns=["rationale"], inplace=True)
    concat_dataset = pd.concat([vstar_dataset, chart_dataset, thinklite_dataset])
    concat_dataset = datasets.Dataset.from_pandas(concat_dataset)

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

    concat_dataset = concat_dataset.map(function=process_fn, with_indices=True, num_proc=8)
    
    # Split dataset: 2k for validation, rest for training
    train_test_split = concat_dataset.train_test_split(test_size=1000, seed=42)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]
    
    # Save train and validation datasets
    train_dataset.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.save_dir, "val.parquet"))
