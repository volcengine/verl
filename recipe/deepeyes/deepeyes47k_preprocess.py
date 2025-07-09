"""
Preprocess the DeepEyes dataset.

We should add some extra_info to use verl's multi-turn function calling.
"""

import argparse
import base64
import io
import os

import datasets
from datasets import load_dataset
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="path/to/local/dir")
    parser.add_argument("--save_dir", default="path/to/save/dir")
    args = parser.parse_args()
    data_source = "hiyouga/DeepEyes-Datasets-47k"

    dataset = load_dataset(
        path=args.dataset_dir,
        data_files=["data_0.1.2_visual_toolbox_v2.parquet"],
    )["train"]

    train_test_split = dataset.train_test_split(test_size=1000, seed=42)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": ("You are a helpful assistant."),
                    },
                    {
                        "role": "user",
                        "content": example["prompt"][1]["content"],
                    },
                ],
                "images": [Image.open(io.BytesIO(image["bytes"])) for image in example["images"]],
                "ability": example["ability"],
                "reward_model": example["reward_model"],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": example["reward_model"]["ground_truth"],
                    "question": example["prompt"][1]["content"],
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_zoom_in_tool": {
                            "create_kwargs": {
                                "image": "data:image/jpeg;base64,"
                                + base64.b64encode(example["images"][0]["bytes"]).decode("utf-8")
                            },
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    train_dataset = train_dataset.cast_column("images", datasets.Sequence(datasets.Image()))

    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True, num_proc=8)
    val_dataset = val_dataset.cast_column("images", datasets.Sequence(datasets.Image()))

    # Save train and validation datasets
    train_dataset.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.save_dir, "val.parquet"))
