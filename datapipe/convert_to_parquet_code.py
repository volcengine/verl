import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse

from system_prompt import LCB_SYSTEM_MESSAGE_GENERIC, LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE, LCB_FORMATTING_WITHOUT_STARTER_CODE
from handle_array import process_reward_model_inputs

def fetch_live_code_bench_system_prompt(question: str, starter_code: str = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt= LCB_SYSTEM_MESSAGE_GENERIC + "\n\n"
    prompt += f"### Question: {question}\n\n"
    if starter_code:
        prompt += (
                f"\n\n### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"\n\n### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def handle_array(answer, starter_code):
    if starter_code and answer.get("fn_name", None) is not None:
        return answer
    processed_inputs = process_reward_model_inputs(answer["inputs"])
    return {
        "inputs": processed_inputs,
        "outputs": answer["outputs"],
        "fn_name": answer.get("fn_name", None)
    }

def load_local_dataset(file_path):
    """Load a local dataset from a jsonl file."""
    if not os.path.exists(file_path):
        print(f"Warning: Local dataset file {file_path} does not exist")
        return datasets.Dataset.from_dict({"problem": [], "solution": [], "starter_code": []})

    data = {"question": [], "answer": [], "starter_code": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data["question"].append(item["question"])
                test_cases = {
                    "inputs": [t.strip().strip('"') for t in item["test_cases"]["inputs"]],
                    "outputs": [t.strip().strip('"') for t in item["test_cases"]["outputs"]],
                    "fn_name": item["test_cases"].get("fn_name", None)
                }
                data["answer"].append(test_cases)
                data["starter_code"].append(item["extra_params"].get("starter_code", None))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")

    return datasets.Dataset.from_dict(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/share/reasoning')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--local_train_dataset', default='/home/liunazhou/data/rl_code_train_0701_test.jsonl')
    parser.add_argument('--local_benchmark_dataset', default='/home/liunazhou/data/rl_code_benchmark_0701_test.jsonl')

    args = parser.parse_args()

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

            starter_code = example.get("starter_code", None)
            prompt = fetch_live_code_bench_system_prompt(question, starter_code)
            answer = handle_array(answer, starter_code)

            data = {
                "data_source": "code",
                "prompt": [{
                    "role": "user",
                    "content": prompt
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

    train_dataset = load_local_dataset(args.local_train_dataset)
    benchmark_dataset = load_local_dataset(args.local_benchmark_dataset)
    train_dataset = train_dataset.map(function=train_make_map_fn('local'), with_indices=True)
    benchmark_dataset = benchmark_dataset.map(function=train_make_map_fn('local'), with_indices=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Benchmark dataset size: {len(benchmark_dataset)}")
    # Print a sample from the processed dataset
    sample_idx = 6520  # You can change this to view different examples
    print("\n===== SAMPLE FROM PROCESSED DATASET =====")
    print(f"Sample index: {sample_idx}")
    sample = train_dataset[sample_idx]
    print(f"Prompt: {sample['prompt'][0]['content']}")  # Show first 200 chars of prompt
    print(f"Data source: {sample['data_source']}")
    print(f"Ability: {sample['ability']}")
    if 'ground_truth' in sample['reward_model']:
        print(f"Ground truth: {sample['reward_model']['ground_truth']}")
    print("==========================================\n")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'rl_code_train_0701_test.parquet'))
    benchmark_dataset.to_parquet(os.path.join(local_dir, 'rl_code_benchmark_0701_test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
