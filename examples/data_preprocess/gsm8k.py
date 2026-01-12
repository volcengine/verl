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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets
import ray

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


@ray.remote(num_cpus=0)
def download_and_preprocess_on_node(data_source, local_dataset_path, local_save_dir, hdfs_dir):
    """Download and preprocess the GSM8k dataset on a specific node."""
    import socket

    hostname = socket.gethostname()
    print(f"Downloading and preprocessing GSM8k dataset on node: {hostname}")

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Expand ~ in the path
    local_save_dir_expanded = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir_expanded, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir_expanded, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir_expanded, "test.parquet"))

    print(f"Dataset saved to {local_save_dir_expanded} on node: {hostname}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir_expanded, dst=hdfs_dir)
        print(f"Dataset copied to HDFS: {hdfs_dir} from node: {hostname}")

    return hostname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Download to all nodes in the Ray cluster instead of just head node."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    if args.distributed:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Get all available nodes and schedule tasks on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        print(f"Found {len(alive_nodes)} alive nodes in the Ray cluster:")
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            print(f"  - {node_ip}")

        # Download and preprocess on all nodes
        download_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = download_and_preprocess_on_node.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(data_source, local_dataset_path, local_save_dir, hdfs_dir)
            download_tasks.append(task)

        # Wait for all tasks to complete
        completed_nodes = ray.get(download_tasks)
        print(f"\nDataset successfully downloaded and preprocessed on {len(completed_nodes)} nodes:")
        for hostname in completed_nodes:
            print(f"  - {hostname}")
    else:
        # Original single-node behavior
        if local_dataset_path is not None:
            dataset = datasets.load_dataset(local_dataset_path, "main")
        else:
            dataset = datasets.load_dataset(data_source, "main")

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        instruction_following = 'Let\'s think step by step and output the final answer after "####".'

        # add a row to each data item that represents a unique id
        def make_map_fn(split):
            def process_fn(example, idx):
                question_raw = example.pop("question")

                question = question_raw + " " + instruction_following

                answer_raw = example.pop("answer")
                solution = extract_solution(answer_raw)
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "answer": answer_raw,
                        "question": question_raw,
                    },
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

        local_save_dir_expanded = os.path.expanduser(local_save_dir)
        os.makedirs(local_save_dir_expanded, exist_ok=True)

        train_dataset.to_parquet(os.path.join(local_save_dir_expanded, "train.parquet"))
        test_dataset.to_parquet(os.path.join(local_save_dir_expanded, "test.parquet"))

        if hdfs_dir is not None:
            makedirs(hdfs_dir)

            copy(src=local_save_dir_expanded, dst=hdfs_dir)
