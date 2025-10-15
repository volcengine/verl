# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

# prepare eval dataset including AIME'24, AIME'25

# hf download math-ai/aime24 --repo-type dataset --local-dir /opt/tiger/datasets/math-ai/aime24
# hf download math-ai/aime25 --repo-type dataset --local-dir /opt/tiger/datasets/math-ai/aime25

import os

import datasets

from verl.utils.reward_score.math_reward import remove_boxed

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


def make_map_fn(data_source):
    def process_fn(example, idx):
        question_raw = example.pop("problem")

        question = question_raw + " " + instruction_following

        if "solution" not in example:
            example["solution"] = example["answer"]

        answer_raw = example.pop("solution")

        example.clear()

        try:
            solution = remove_boxed(answer_raw)
        except Exception:
            solution = answer_raw

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
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math-ai", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    if args.local_dataset_path is not None:
        aime24_dataset_path = os.path.join(args.local_dataset_path, "math-ai/aime24")
        aime25_dataset_path = os.path.join(args.local_dataset_path, "math-ai/aime25")
    else:
        aime24_dataset_path = "math-ai/aime24"
        aime25_dataset_path = "math-ai/aime25"

    aime24_dataset = datasets.load_dataset(aime24_dataset_path, split="test")
    aime25_dataset = datasets.load_dataset(aime25_dataset_path, split="test")

    aime24_dataset = aime24_dataset.map(function=make_map_fn("aime24"), with_indices=True)
    aime25_dataset = aime25_dataset.map(function=make_map_fn("aime25"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    aime24_dataset.to_parquet(os.path.join(local_save_dir, "aime24_test.parquet"))
    aime25_dataset.to_parquet(os.path.join(local_save_dir, "aime25_test.parquet"))
