import os
import datasets
import json

import argparse
from transformers import AutoTokenizer

import base64
import json
import pickle
import zlib
from datetime import datetime


def filter_date(dataset, start_date=None, end_date=None):
    new_dataset = []

    for item in dataset:
        contest_date = datetime.fromisoformat(item['contest_date'])
        if start_date is not None:
            p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if p_start_date > contest_date:
                continue

        if end_date is not None:
            p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            if p_end_date < contest_date:
                continue

        new_dataset.append(item)

    if start_date or end_date:
        print(
            f'Filtered dataset with start_date: {start_date}, end_date: {end_date}, remaining items: {len(new_dataset)}'
        )
    return new_dataset


def load_local_dataset(file_path):
    """Load a local dataset from a jsonl file."""
    if not os.path.exists(file_path):
        print(f"Warning: Local dataset file {file_path} does not exist")
        return datasets.Dataset.from_dict({"problem": [], "solution": []})

    dataset = []
    data = {"question": [], "answer": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # load test cases
                public_test_cases = item['public_test_cases']
                public_test_cases = json.loads(item['public_test_cases'])

                private_test_cases = item['private_test_cases']
                try:
                    private_test_cases = json.loads(item['private_test_cases'])
                except Exception as e:  # noqa: F841
                    private_test_cases = json.loads(
                        pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode('utf-8'))  # type: ignore
                                                    )))  # type: ignore
                
                metadata = json.loads(item['metadata'])
                evaluation_sample = {
                    'inputs': [t['input'].strip().strip('"') for t in public_test_cases + private_test_cases],
                    'outputs': [t['output'].strip().strip('"') for t in public_test_cases + private_test_cases],
                }
                item['evaluation_sample'] = evaluation_sample

                dataset.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")

    new_dataset = filter_date(dataset, "2024-10-01", "2025-01-31")

    for item in new_dataset:
        data["question"].append(item["question_content"])
        data["answer"].append(item["evaluation_sample"])

    return datasets.Dataset.from_dict(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/share/reasoning')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--local_dataset', default='/home/share/reasoning/livecodebench/test5.jsonl')

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
            
            # user_prompt = "{question}\n\nPresent the code in \n```python\nYour code\n```\nat the end.\nThe code must take input from Standard Input and print answer."
            user_prompt = "{question}"

            data = {
                "data_source": "code",
                "prompt": [{
                    "role": "user",
                    "content": user_prompt.format(question=question)
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
    sample_idx = 1  # You can change this to view different examples
    print("\n===== SAMPLE FROM PROCESSED DATASET =====")
    print(f"Sample index: {sample_idx}")
    sample = train_dataset[sample_idx]
    print(f"Prompt: {sample['prompt'][0]['content']}")  # Show first 200 chars of prompt
    print(f"Token length: {len(tokenizer.encode(sample['prompt'][0]['content']))}")
    print(f"Data source: {sample['data_source']}")
    print(f"Ability: {sample['ability']}")
    # if 'ground_truth' in sample['reward_model']:
    #     print(f"Ground truth: {sample['reward_model']['ground_truth']}")
    print("==========================================\n")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'code_test.parquet'))
