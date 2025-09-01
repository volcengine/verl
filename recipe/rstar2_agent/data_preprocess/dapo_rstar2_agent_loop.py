# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Preprocess the DAPO-Math-17k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/rstar2-agent/dapo-math-17k-en")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "open-r1/DAPO-Math-17k-Processed"
    dataset = datasets.load_dataset(data_source, "en")

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
            solution = example["reward_model"]["ground_truth"]

            data = {
                "data_source": f"rstar_{data_source}",
                "agent_name": "rstar2_agent",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                    "need_tools_kwargs": False,
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": solution,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
