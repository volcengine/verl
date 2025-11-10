"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import argparse
import os
import re
from random import choice, randint, seed
from typing import List, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        
        samples.append((target, numbers))
    
    return samples

def make_prefix(dp, template_type):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/MoPPS/data/countdown3to4')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=512)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'countdown'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    
    try:
        raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    except:
        local_data_source = f"{args.local_dir}/raw/data"
        if not os.path.exists(local_data_source):
            os.system('pip install -U huggingface_hub; export HF_ENDPOINT=https://hf-mirror.com; huggingface-cli download --repo-type dataset --resume-download Jiayi-Pan/Countdown-Tasks-3to4 --local-dir data/countdown3to4/raw')
        if os.path.exists(local_data_source):
            raw_dataset = load_dataset('parquet', data_files={
                "train": os.path.join(local_data_source, "train-00000-of-00001.parquet")
            })
            raw_dataset = raw_dataset['train']
        else:
            raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['target'],
                "numbers": example['nums']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 