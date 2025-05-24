# SPDX-FileCopyrightText: (c) iSE UIUC Research Group
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule
import rich

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.coder1 import code_exec, exec_check_stdio, _ERROR_MSG_PREFIX

N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"


# this dataset is super noisy and needs code execution to verify the tasks
def taco():
    rich.print(Rule("Loading likaixin/TACO-verified..."))
    dataset = load_dataset("likaixin/TACO-verified")["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return _EMPTY_RETURN_

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return _EMPTY_RETURN_

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return _EMPTY_RETURN_

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                prompt_pieces.append("Also feel free to reuse/extend the following starter code:")
                prompt_pieces.append(f"```python\n{example['starter_code'].strip()}\n```")

            ##
            ## Customization
            ##
            if "fn_name" in oracle:  # the dataset is too noisy
                fn_name = oracle["fn_name"]
                if source == "leetcode":
                    fn_name = "Solution()." + fn_name

                test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""

                if source in ["leetcode", "hackerrank"]:
                    test_code += f"    assert _deep_eq({fn_name}(*i), o)"
                elif source == "codewars":
                    test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
                else:
                    raise ValueError(f"Unknown source: {source}")

                _check_test = example["solutions"][-1] + "\n" + test_code
                if source in ["leetcode"]:
                    _check_test = PY_IMPORTS + _check_test

                succ, err = code_exec(_check_test)
                if not succ:
                    rich.print(f"[bold red]Test code failed for {source}")
                    print(_check_test)
                    print(err)
                    return _EMPTY_RETURN_
                oracle = json.dumps({"functional": test_code})
                assert example["starter_code"].strip() != ""
            elif "inputs" in oracle and "outputs" in oracle:
                stdin_list, stdout_list = minimize_stdio(oracle["inputs"], oracle["outputs"])
                if len(stdin_list) == 0:
                    return _EMPTY_RETURN_

                with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(executor.submit(exec_check_stdio, example["solutions"][-1], stdin, stdout))
                    for future in as_completed(futures):
                        exec_succ, output, stdin, stdout = future.result()
                        pass_test = exec_succ and output.strip() == stdout.strip()
                        if not pass_test:
                            rich.print(f"[bold red]Test code failed for {source}")
                            print(example["solutions"][-1])
                            print(f"{exec_succ = }")
                            print(f"{stdin = }", f"{stdout = }")
                            if output.startswith(_ERROR_MSG_PREFIX):
                                print("output = \n", output)
                            else:
                                print(f"{output = }")
                            return _EMPTY_RETURN_

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)
            return {
                "data_source": "code",
                "prompt": [{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, {
                    "role": "user",
                    "content": prompt
                }],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (example["solutions"][0] if example["solutions"] else ""),
                    "dataset": "likaixin/TACO-verified",
                    "timeout": 10,  # seconds
                },
            }

        return process_fn

    dataset = dataset.map(function=make_map_fn("train"),
                          with_indices=True,
                          num_proc=64,
                          remove_columns=dataset.column_names).filter(lambda x: x != _EMPTY_RETURN_)
    splits = dataset.train_test_split(test_size=max(1, min(N_TESTSET_PER_DATASET, len(dataset) * 0.1)), seed=666)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    for t in dataset:
        print(f"{t = }")
        t["extra_info"]["split"] = "test"

    return train_dataset, test_dataset


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    test_dataset = load_dataset("json",
                                data_files="LeetCodeDataset/data/LeetCodeDataset-v2-test-problems.jsonl")["train"]
    print("Test set:", test_dataset)

    train_dataset = concatenate_datasets([
        load_dataset(
            "json",
            data_files="LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl",
        )["train"],
        load_dataset(
            "json",
            data_files="LeetCodeDataset/data/LeetCodeDataset-v2-sft-problems.jsonl",
        )["train"],
    ]).filter(
        lambda example: example["meta"]["question_id"] not in set([d["question_id"] for d in test_dataset["meta"]]))
    print("Before deduplication - Training set:", train_dataset)

    first_time_idx = []
    seen_question_ids = set()
    for i, example in enumerate(train_dataset):
        if example["meta"]["question_id"] not in seen_question_ids:
            first_time_idx.append(i)
            seen_question_ids.add(example["meta"]["question_id"])
    train_dataset = train_dataset.select(first_time_idx)

    print("After deduplication - Training set:", train_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = f"Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{example['meta']['query'].strip()}"
            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style":
                        "rule",
                    "ground_truth":
                        json.dumps({"functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})"}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],  # C++?
                    "prompt": prompt,
                    "dataset": "LeetCodeDataset",
                    "timeout": 5,  # seconds
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./data/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    subprocess.run(["git", "clone", "https://github.com/newfacade/LeetCodeDataset.git"])
    subprocess.run(["git", "-C", "LeetCodeDataset", "checkout", "f3519e6"])

    root_dir = args.root_dir
    hdfs_dir = args.hdfs_dir

    train_datasets = []
    test_datasets = []

    dataset_makes = [leetcode2k] if args.test else [leetcode2k, taco]
    names = "-".join([make.__name__ for make in dataset_makes])

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(root_dir, f"code-r1-{round(len(train_dataset) / 1000)}k-{names}")
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=root_dir, dst=hdfs_dir)
