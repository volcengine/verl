# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess DAPO dataset to parquet format
"""

import argparse
import os

import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/dapo-with-aime24")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--dapo_dataset_path", type=str, default="ftajwar/deduplicated_dapo_dataset")
    parser.add_argument("--aime24_part_1_dataset_path", type=str, default="MathArena/aime_2024_I")
    parser.add_argument("--aime24_part_2_dataset_path", type=str, default="MathArena/aime_2024_II")
    parser.add_argument("--train_size", type=int, default=4096)

    args = parser.parse_args()

    data_source = "math_dapo"

    # Load DAPO dataset for training
    dapo_dataset_path = args.dapo_dataset_path
    dapo_dataset = datasets.load_dataset(dapo_dataset_path, trust_remote_code=True)

    # Load AIME 2024 part 1 dataset for testing
    aime24_dataset_path_part_1 = args.aime24_part_1_dataset_path
    aime24_dataset_part_1 = datasets.load_dataset(aime24_dataset_path_part_1, trust_remote_code=True)

    # Load AIME 2024 part 2 dataset for testing
    aime24_dataset_path_part_2 = args.aime24_part_2_dataset_path
    aime24_dataset_part_2 = datasets.load_dataset(aime24_dataset_path_part_2, trust_remote_code=True)

    train_dataset = dapo_dataset["train"]
    train_dataset = train_dataset.select(np.random.choice(len(train_dataset), size=args.train_size, replace=False))

    dev_dataset_aime24_part_1 = aime24_dataset_part_1["train"]
    dev_dataset_aime24_part_2 = aime24_dataset_part_2["train"]
    dev_dataset = datasets.concatenate_datasets([dev_dataset_aime24_part_1, dev_dataset_aime24_part_2])

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            if "prompt" in example:
                question = example.pop("prompt")
            elif "problem" in example:
                question = example.pop("problem")
            else:
                raise ValueError(f"Unknown question type: {example}")

            question = question + " " + instruction_following

            if "answer" in example:
                solution = example.pop("answer")
            else:
                raise ValueError(f"Unknown answer type: {example}")
            solution = str(solution)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(local_dir, "dev.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
