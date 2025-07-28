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
Preprocess a huggingface/benchmax dataset to a multiturn format suitable for a benchmax environment.
"""

import argparse
import os
from importlib import import_module
from types import ModuleType
from typing import Type

import datasets

from benchmax.envs.base_env import BaseEnv
from verl.utils.hdfs_io import copy, makedirs

def load_class(dotted_path: str) -> BaseEnv:
    """
    Load and return the class specified by `dotted_path`.
    Example: "benchmax.envs.wikipedia.wiki_env.WikipediaEnv"
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as exc:
        raise ImportError(
            f'"{dotted_path}" doesn’t look like "package.module.Class"'
        ) from exc

    module: ModuleType = import_module(module_path)
    try:
        cls: BaseEnv = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f'Module "{module_path}" has no attribute "{class_name}"'
        ) from exc

    return cls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Local directory where processed train/test parquet files will be written."
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Identifier of the HuggingFace dataset to load (e.g., 'squad', 'wikitext')."
    )
    parser.add_argument(
        "--env_path",
        required=True,
        help=(
            "Dotted path to the BaseEnv subclass to use for preprocessing, "
            "e.g. 'benchmax.envs.wikipedia.wiki_env.WikipediaEnv'."
        )
    )
    parser.add_argument(
        "--prompt_key",
        default="prompt",
        help=(
            "Name of the field in the dataset examples that contains the user prompt. "
            "Defaults to 'prompt'."
        )
    )
    parser.add_argument(
        "--ground_truth_key",
        default="ground_truth",
        help=(
            "Name of the field in the dataset examples that contains the expected response. "
            "If omitted or set to an empty string, no ground-truth field will be included."
        )
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help=(
            "Optional HDFS target directory. If provided, the output local_dir will be "
            "copied there after processing."
        )
    )

    args = parser.parse_args()

    print(f"Loading {args.dataset_name} dataset...", flush=True)
    benchmax_cls: Type[BaseEnv] = load_class(args.env_path)
    dataset, dataset_path = benchmax_cls.load_dataset(args.dataset_name)
    benchmax_env: BaseEnv = benchmax_cls(dataset_path=dataset_path)
    tool_names = [t.name for t in benchmax_env.list_tools()]
    dataset = dataset.map(
        lambda example: benchmax_env.dataset_preprocess(example),
    )
    if isinstance(dataset, dict) and set(dataset.keys()) == set(["train", "test"]):
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    else:
        if isinstance(dataset, dict):
            dataset = datasets.concatenate_datasets(
                [ds for ds in dataset.values()]
            ).shuffle(seed=42)
        split = dataset.train_test_split(
            test_size=0.2, seed=42, shuffle=False
        )
        train_dataset = split["train"]
        test_dataset  = split["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = [
                {"role": "system", "content": benchmax_cls.system_prompt},
                {"role": "user", "content": example[args.prompt_key]}
            ]
            example["prompt"] = prompt
            if args.ground_truth_key:
                example["ground_truth"] = example[args.ground_truth_key]
            # Add everything else to extra_info
            extra_info = {
                k: v for k, v in example.items()
                if k not in [
                    args.prompt_key, args.ground_truth_key,
                    "init_rollout_args"
                ]
            }
            create_args = example.get("init_rollout_args", {}) or {"dummy": "dummy"}
            extra_info["tools_kwargs"] = {
                tool_name: {
                    "create_kwargs": {**create_args},
                } for tool_name in tool_names
            }

            example.pop("init_rollout_args")
            # This extra_info is used to pass addition info during reward computation
            example["extra_info"] = extra_info
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
