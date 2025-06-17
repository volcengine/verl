# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Retool dataset to parquet format
"""

import argparse
import os

import pandas as pd
import datasets

from verl.utils.hdfs_io import copy, makedirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/retool_multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_ratio", default=0.9, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    data_source = "/user/longxiang1/data/swordfaith/ReTool-SFT-multi-turn"
    dataset = datasets.load_dataset(data_source, "default")
    data_source = "swordfaith/ReTool-SFT-multi-turn"

    train_dataset = dataset["train"]
    shuffled_train_dataset = train_dataset.shuffle(seed=args.seed)
    split_idx = int(len(shuffled_train_dataset) * args.train_ratio)
    train_dataset = shuffled_train_dataset.select(range(split_idx))
    test_dataset = shuffled_train_dataset.select(range(split_idx, len(shuffled_train_dataset)))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            messages = example.pop("messages")
            tools = example.pop("tools")
            data = {
                "data_source": data_source,
                "messages": messages,
                "tools": tools,
                "enable_thinking": False,
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
