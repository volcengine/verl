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
Preprocess the docleaderboard-queries dataset to parquet format
"""

import argparse
import csv
import os

import datasets


def load_queries_from_tsv(file_path):
    """Load queries from TSV file and return as list of dictionaries."""
    queries = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries.append({"query_text": row[1]})
    return queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="./data/docleaderboard-queries.tsv",
        help="Path to the input TSV file",
    )
    parser.add_argument("--local_dir", default="~/data/docleaderboard-queries")
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Fraction of data to use for training"
    )

    args = parser.parse_args()

    # Expand paths
    input_file = os.path.expanduser(args.input_file)
    local_dir = os.path.expanduser(args.local_dir)

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Load queries from TSV
    queries = load_queries_from_tsv(input_file)

    data_source = "docleaderboard-queries"

    # Process queries into the expected format
    processed_data = []
    instruction = "Reformulate this search query to be a shorter and more explicit query."

    for idx, query_item in enumerate(queries):
        query_text = query_item["query_text"]

        # Create prompt with instruction
        prompt = f"Given the following query {query_text}, please reason about the query and reformulate it to be a shorter and more explicit query that is more likely to lead to a relevant result."

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "search_reformulation",
            "reward_model": {"style": "rule"},
            "extra_info": {
                "split": "train" if idx < len(queries) * args.train_split else "test",
                "index": idx,
                "question": query_text,
            },
        }
        processed_data.append(data)

    # Split into train and test
    split_idx = int(len(processed_data) * args.train_split)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    # Convert to datasets
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    # Save as parquet files
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Processed {len(queries)} queries:")
    print(f"  Train set: {len(train_data)} queries")
    print(f"  Test set: {len(test_data)} queries")
    print(f"  Saved to: {local_dir}")
