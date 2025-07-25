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
Preprocess the Text to Speech dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Path to training JSON/JSONL file")
    parser.add_argument("--test_file", required=True, help="Path to test JSON/JSONL file")
    parser.add_argument("--local_dir", default=None, required=True)
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Load datasets from local JSON files
    train_dataset = datasets.load_dataset("json", data_files=args.train_file)['train']
    test_dataset = datasets.load_dataset("json", data_files=args.test_file)['train']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            text = example.pop("text")

            # use cosyvoice2 official huggingface compatible checkpoint template
            question = text
            answer = ""

            data = {
                "data_source": f"{args.train_file}_{args.test_file}",  # Use file names as data source
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ],
                "ability": "text-to-speech",
                "reward_model": {"style": "rule", "ground_truth": text},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "text": text,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    print(train_dataset)
    print(test_dataset)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
