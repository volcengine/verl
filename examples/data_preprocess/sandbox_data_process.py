# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess a huggingface dataset to a multiturn format suitable for a sandbox environment.
"""

import argparse
import os
from importlib import import_module
from types import ModuleType
from typing import Type, Any

import datasets

from envs.base_sandbox import BaseSandbox
from verl.utils.hdfs_io import copy, makedirs

def load_class(dotted_path: str) -> BaseSandbox:
    """
    Load and return the class specified by `dotted_path`.
    Example: "envs.wikipedia.WikipediaSandbox"
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as exc:          # no dot in the path
        raise ImportError(
            f'"{dotted_path}" doesn’t look like "package.module.Class"'
        ) from exc

    module: ModuleType = import_module(module_path)
    try:
        cls: BaseSandbox = getattr(module, class_name)
    except AttributeError as exc:      # class not found inside module
        raise ImportError(
            f'Module "{module_path}" has no attribute "{class_name}"'
        ) from exc

    return cls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--sandbox_path", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--ground_truth_key", default="")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    print(f"Loading the {args.dataset_path} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(args.dataset_path, trust_remote_code=True)
    if set(dataset.keys()) == set(["train", "test"]):
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    else:
        full_dataset = datasets.concatenate_datasets(
            [ds for ds in dataset.values()]
        ).shuffle(seed=42)
        split = full_dataset.train_test_split(
            test_size=0.2, seed=42, shuffle=False
        )
        train_dataset = split["train"]
        test_dataset  = split["test"]
    sandbox_cls: BaseSandbox = load_class(args.sandbox_path)

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = [
                {"role": "system", "content": sandbox_cls.system_prompt},
                {"role": "user", "content": example[args.prompt_key]}
            ]
            example["prompt"] = prompt
            if args.ground_truth_key:
                example["ground_truth"] = example[args.ground_truth_key]
            return example
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
