import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/share/reasoning')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'MatrixStudio/Codeforces-Python-Submissions'

    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = """\n\nLets think step-by-step and provide the final Python code solution to the problem. Make sure read inputs using input() directly and print the result using print() directly.
    """

    def make_map_fn(split):

        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": example['prompt'] + instruction_following 
                }],
                "ability": "code",
                "programming_language": example['programmingLanguage'],
                "time_limit": example['time-limit'],
                "memory_limit": example['memory-limit'],
                "points": example['points'],
                #put the test cases in the ground truth
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "inputs": [test_case['input'].strip().strip('"') for test_case in example['test_cases']],
                        "outputs": [test_case['output'].strip().strip('"') for test_case in example['test_cases']],
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    def remove_duplicates(dataset):
        unique_prompts = set()
        unique_indices = []
        
        # Find the indices of unique prompts
        for i, prompt in enumerate(dataset['prompt']):
            simplified_prompt = ' '.join(prompt[0]['content'].lower().split())
            if simplified_prompt not in unique_prompts:
                unique_prompts.add(simplified_prompt)
                unique_indices.append(i)
        
        return dataset.select(unique_indices)

    train_dataset = remove_duplicates(train_dataset)
    # test_dataset = remove_duplicates(test_dataset)

    print(f"After deduplication: Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained("/home/share/reasoning/DeepSeek-R1-Distill-Qwen-7B")

    max_token_length = 2048
    
    def filter_by_token_length(example):
        question = example['prompt'][0]['content']
        token_length = len(tokenizer.encode(question))
        return token_length <= max_token_length

    def filter_by_test_cases_length(example):
        test_cases = example['reward_model']['ground_truth']
        inputs = test_cases['inputs']
        outputs = test_cases['outputs']
        return len(outputs) < 82

    train_dataset = train_dataset.filter(filter_by_token_length)
    # test_dataset = test_dataset.filter(filter_by_token_length)

    train_dataset = train_dataset.filter(filter_by_test_cases_length)
    # test_dataset = test_dataset.filter(filter_by_test_cases_length)

    # combined_dataset = datasets.concatenate_datasets([train_dataset, test_dataset])
    # combined_dataset = train_dataset
    # print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Shuffle the combined dataset for better distribution
    # combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Split into new train and test sets
    # train_size = min(6000, len(combined_dataset) - 100)  # Ensure we have at least some test examples
    # train_dataset = combined_dataset.select(range(train_size))
    # test_dataset = combined_dataset.select(range(train_size, len(combined_dataset)))
    
    # Update the split information in extra_info
    def update_split_info(example, split_name):
        example['extra_info']['split'] = split_name
        return example
    
    # train_dataset = train_dataset.map(lambda x: update_split_info(x, 'train'))
    # test_dataset = test_dataset.map(lambda x: update_split_info(x, 'test'))

    print(f"Final train dataset size: {len(train_dataset)}")
    # print(f"Final test dataset size: {len(test_dataset)}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'rl_code_data_codeforces.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
