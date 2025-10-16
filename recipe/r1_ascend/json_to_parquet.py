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

import argparse
import json
import os
import random

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="deepscaler.json to parquet file", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output_dir", type=str, required=True, help="output dir for train/test parquet file")
    parser.add_argument("--json_path", type=str, default="./deepscaler.json", help="path of deepscaler.json")
    parser.add_argument("--train_data_ratio", type=float, default=0.9, help="ratio of train data")
    return parser.parse_args()


def validate_arguments(args):
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"File not found: {args.json_path}")
    if not 0 < args.train_data_ratio < 1:
        raise ValueError("Train data ratio should be between 0 and 1")
    os.makedirs(args.output_dir, exist_ok=True)


def convert_json_to_parquet(json_path, train_data_ratio, output_dir):
    random.seed(42)

    with open(json_path, encoding="utf-8") as f:
        original_data = json.load(f)

    converted_data = []
    for item in original_data:
        r1_template = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within </think> </think> "
            "and </think> </think> tags, respectively, i.e., "
            "</think> reasoning process here </think></think> answer here </think>. "
            "Put your final answer within \\boxed{{}}. <｜User｜>{content}<｜Assistant｜>"
        )
        content_with_template = r1_template.format(content=item["problem"])
        converted_item = {
            "data_source": "deepscaler",
            "prompt": [{"content": content_with_template, "role": "user"}],
            "ability": "math",
            "reward_model": {"ground_truth": item["answer"], "style": "rule"},
            "extra_info": {"answer": item["solution"]},
        }
        converted_data.append(converted_item)

    split_index = int(len(converted_data) * train_data_ratio)
    train_data = converted_data[:split_index]
    test_data = converted_data[split_index:]

    for item in train_data:
        item["split"] = "train"
    for item in test_data:
        item["split"] = "test"
    all_data = train_data + test_data
    df = pd.DataFrame(all_data)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    del train_df["split"]
    del test_df["split"]

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), engine="pyarrow", index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), engine="pyarrow", index=False)
    print(
        f"Json to parquet success! Total num {len(all_data)}, train num {len(train_data)}, test num {len(test_data)}",
        flush=True,
    )


def main():
    try:
        args = parse_arguments()
        validate_arguments(args)
        convert_json_to_parquet(args.json_path, args.train_data_ratio, args.output_dir)
    except Exception as e:
        print(f"[ERROR]: {e}")
        exit(1)


if __name__ == "__main__":
    main()
