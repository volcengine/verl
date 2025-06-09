import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse
from transformers import AutoTokenizer
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def load_local_dataset(file_path):
    """Load a local dataset from a jsonl file."""
    if not os.path.exists(file_path):
        print(f"Warning: Local dataset file {file_path} does not exist")
        return datasets.Dataset.from_dict({"problem": [], "solution": []})

    data = {"question": [], "answer": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data["question"].append(item["question"] + "\nThe code must take input from Standard Input and print answer.")
                test_cases = {
                    "inputs": [t.strip().strip('"') for t in item["test_cases"]["inputs"]],
                    "outputs": [t.strip().strip('"') for t in item["test_cases"]["outputs"]]
                }
                data["answer"].append(test_cases)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")

    return datasets.Dataset.from_dict(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/share/reasoning')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--local_dataset', default='/home/share/reasoning/rl_code_data_0528.jsonl')

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("/home/share/reasoning/DeepSeek-R1-Distill-Qwen-7B")

    # add a row to each data item that represents a unique id
    def train_make_map_fn(split):

        def process_fn(example, idx):
            if split == 'local':
                question = example.pop('question')
                answer = example.pop('answer')
            else:
                question = example.pop('problem')
                solution = example.pop('solution')
                answer = extract_answer(solution)

            data = {
                "data_source": "code",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            } 
            return data

        return process_fn

    train_dataset = load_local_dataset(args.local_dataset)
    train_dataset = train_dataset.map(function=train_make_map_fn('local'), with_indices=True)
    print(f"Loaded local dataset with {len(train_dataset)} examples")

    max_token_length = 2048

    def filter_by_token_length(example):
        question = example['prompt'][0]['content']
        token_length = len(tokenizer.encode(question))
        return token_length <= max_token_length

    # Filter the datasets
    train_dataset = train_dataset.filter(filter_by_token_length)

    print(f"Train dataset size: {len(train_dataset)}")

    # Print a sample from the processed dataset
    sample_idx = 6520  # You can change this to view different examples
    print("\n===== SAMPLE FROM PROCESSED DATASET =====")
    print(f"Sample index: {sample_idx}")
    sample = train_dataset[sample_idx]
    print(f"Prompt: {sample['prompt'][0]['content']}")  # Show first 200 chars of prompt
    print(f"Token length: {len(tokenizer.encode(sample['prompt'][0]['content']))}")
    print(f"Data source: {sample['data_source']}")
    print(f"Ability: {sample['ability']}")
    if 'ground_truth' in sample['reward_model']:
        print(f"Ground truth: {sample['reward_model']['ground_truth']}")
    print("==========================================\n")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'rl_code_data_0528.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
