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
Preprocess OpenHands SFT Trajectories dataset into parquet format for multi-turn training
"""

import os
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs
from transformers import AutoTokenizer


def count_tokens(text, tokenizer):
    """Count the number of tokens in a text"""
    return len(tokenizer(text).input_ids)


def process_conversation(example, idx, split, tokenizer, max_tokens=32000):
    """Convert a conversation into the expected format"""
    messages = []
    total_tokens = 0

    # Add system message
    system_msg = {"role": "system", "content": "You are a helpful assistant that can understand and generate code."}
    total_tokens += count_tokens(system_msg["content"], tokenizer)
    messages.append(system_msg)

    # Process each turn
    for i in range(len(example['human'])):
        # Add human message
        human_msg = {"role": "user", "content": example['human'][i]}
        human_tokens = count_tokens(human_msg["content"], tokenizer)

        # Add assistant message
        assistant_msg = {"role": "assistant", "content": example['assistant'][i]}
        assistant_tokens = count_tokens(assistant_msg["content"], tokenizer)

        # Check if adding these messages would exceed token limit
        if total_tokens + human_tokens + assistant_tokens > max_tokens:
            break

        total_tokens += human_tokens + assistant_tokens
        messages.append(human_msg)
        messages.append(assistant_msg)

    # Only return if we have at least one complete turn
    if len(messages) >= 3:  # system + at least one human-assistant pair
        return {
            "data_source": "openhands_sft_trajectories",
            "messages": messages,
            "extra_info": {
                'split': split,
                'index': idx,
                'total_tokens': total_tokens,
                'original_id': example.get('id', None)
            }
        }
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/multiturn')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--max_tokens', type=int, default=32000)

    args = parser.parse_args()

    # Load tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    # Load OpenHands dataset
    dataset = datasets.load_dataset('SWE-Gym/OpenHands-SFT-Trajectories')

    # Split into train/test (90/10 split)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Process the datasets
    train_dataset = train_dataset.map(
        function=lambda x, i: process_conversation(x, i, 'train', tokenizer, args.max_tokens),
        with_indices=True,
        remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(
        function=lambda x, i: process_conversation(x, i, 'test', tokenizer, args.max_tokens),
        with_indices=True,
        remove_columns=test_dataset.column_names)

    # Filter out None values (conversations that were too long)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.filter(lambda x: x is not None)

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    # Print statistics
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Data saved to {local_dir}")
EOF > examples / data_preprocess / multiturn.py
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
Preprocess OpenHands SFT Trajectories dataset into parquet format for multi-turn training
"""

import os
import argparse
import datasets
from verl.utils.hdfs_io import copy, makedirs
from transformers import AutoTokenizer


def count_tokens(text, tokenizer):
    """Count the number of tokens in a text"""
    return len(tokenizer(text).input_ids)


def process_conversation(example, idx, split, tokenizer, max_tokens=32000):
    """Convert a conversation into the expected format"""
    messages = []
    total_tokens = 0

    # Add system message
    system_msg = {"role": "system", "content": "You are a helpful assistant that can understand and generate code."}
    total_tokens += count_tokens(system_msg["content"], tokenizer)
    messages.append(system_msg)

    # Process each turn
    for i in range(len(example['human'])):
        # Add human message
        human_msg = {"role": "user", "content": example['human'][i]}
        human_tokens = count_tokens(human_msg["content"], tokenizer)

        # Add assistant message
        assistant_msg = {"role": "assistant", "content": example['assistant'][i]}
        assistant_tokens = count_tokens(assistant_msg["content"], tokenizer)

        # Check if adding these messages would exceed token limit
        if total_tokens + human_tokens + assistant_tokens > max_tokens:
            break

        total_tokens += human_tokens + assistant_tokens
        messages.append(human_msg)
        messages.append(assistant_msg)

    # Only return if we have at least one complete turn
    if len(messages) >= 3:  # system + at least one human-assistant pair
        return {
            "data_source": "openhands_sft_trajectories",
            "messages": messages,
            "extra_info": {
                'split': split,
                'index': idx,
                'total_tokens': total_tokens,
                'original_id': example.get('id', None)
            }
        }
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/multiturn')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--max_tokens', type=int, default=32000)

    args = parser.parse_args()

    # Load tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    # Load OpenHands dataset
    dataset = datasets.load_dataset('SWE-Gym/OpenHands-SFT-Trajectories')

    # Split into train/test (90/10 split)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Process the datasets
    train_dataset = train_dataset.map(
        function=lambda x, i: process_conversation(x, i, 'train', tokenizer, args.max_tokens),
        with_indices=True,
        remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(
        function=lambda x, i: process_conversation(x, i, 'test', tokenizer, args.max_tokens),
        with_indices=True,
        remove_columns=test_dataset.column_names)

    # Filter out None values (conversations that were too long)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.filter(lambda x: x is not None)

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    # Print statistics
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Data saved to {local_dir}")
