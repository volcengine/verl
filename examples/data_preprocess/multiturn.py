# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/multiturn")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--num_conversations", type=int, default=20)
    parser.add_argument("--num_turns", type=int, default=3)
    args = parser.parse_args()

    # Create example conversations
    conversations = []

    base_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "And what about Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you explain quantum computing?"},
                {
                    "role": "assistant",
                    "content": "Quantum computing uses quantum-mechanical phenomena like superposition and "
                    "entanglement to perform computations.",
                },
                {"role": "user", "content": "How is it different from classical computing?"},
                {
                    "role": "assistant",
                    "content": "Classical computing uses bits that are 0 or 1; quantum computing uses qubits that "
                    "can represent multiple states at once.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a simple Python function to calculate factorial."},
                {
                    "role": "assistant",
                    "content": (
                        "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        "
                        "return n * factorial(n-1)\n```\n"
                    ),
                },
                {"role": "user", "content": "Can you make it iterative instead?"},
                {
                    "role": "assistant",
                    "content": (
                        "```python\ndef factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        "
                        "result *= i\n    return result\n```\n"
                    ),
                },
            ]
        },
    ]
    conversations.extend(base_examples)

    topics = [
        "sorting numbers",
        "time complexity",
        "simple algebra",
        "string processing",
        "prime numbers",
        "geometry basics",
        "unit conversion",
        "probability",
        "date arithmetic",
        "logic puzzles",
    ]
    helpers = [
        "Give a brief answer.",
        "Explain in one sentence.",
        "Show a small example.",
        "List two key points.",
    ]
    total = max(args.num_conversations, len(conversations))
    for idx in range(len(conversations), total):
        topic = topics[idx % len(topics)]
        helper = helpers[idx % len(helpers)]
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for t in range(args.num_turns):
            user = f"Turn {t + 1}: Tell me about {topic}. {helper}"
            assistant = f"Here is a concise note on {topic}. (turn {t + 1})"
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        conversations.append({"messages": messages})

    # Create train and test datasets
    split_idx = max(1, int(len(conversations) * 0.8))
    train_data = conversations[:split_idx]
    test_data = conversations[split_idx:]

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(
        {"messages": pd.Series([item["messages"] for item in train_data], dtype=object)}
    )
    test_df = pd.DataFrame(
        {"messages": pd.Series([item["messages"] for item in test_data], dtype=object)}
    )

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
