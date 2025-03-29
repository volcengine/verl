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
Preprocess the ORZ dataset to parquet format
"""

import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse


PROMPT_TEMPLATE = ("Solve the following math problem step by step. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
                    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. \n\n"
                    "{question}\n\n"
                    "Remember to put your answer inside <answer> </answer> tags, and your final answer will be extracted automatically by the \\boxed{{}} tag.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/orz')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'orz'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    train_dataset = datasets.load_dataset('pe-nlp/ORZ-MATH-57k-Filter-difficulty-decontaminated', split='train')

    # split the dataset into train and test
    test_count = 500
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = train_dataset.select(range(len(train_dataset) - test_count, len(train_dataset)))
    train_dataset = train_dataset.select(range(len(train_dataset) - test_count))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = PROMPT_TEMPLATE.format(question=question)
            solution = example.pop('answer')
            data = {
                "data_source": "lighteval/MATH", # reuse the existing reward function
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "difficulty": max(min(int((1.0-example.pop('pass_at_n')) * 10) + 1, 10), 1),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
