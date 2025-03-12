import os
import datasets
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='setwise-r1')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'Tevatron/msmarco-passage-aug'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)['train'].shuffle(seed=42)

    # select first 100k as training data
    train_dataset = dataset.select(range(100000))
    # select last 1k as test data
    test_dataset = dataset.select(range(len(dataset) - 1000, len(dataset)))
    instruction_following = "Look into each documents carefully. Rank the documents based on their relevance to the user query. Then, pick the most relevant document identifier within \\boxed{}."

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):

            positive_passage = example['positive_passages'][0]['text']
            negative_passages = [passage['text'] for passage in example['negative_passages']]
            random.shuffle(negative_passages)
            negative_passages = negative_passages[:10]

            # concatenate the positive passage and negative passages and shuffle them, keeping the index of the positive passage recorded
            passages = [positive_passage] + negative_passages
            passage_indices = list(range(len(passages)))
            random.shuffle(passage_indices)
            passages = [passages[i] for i in passage_indices]
            target_index = passage_indices.index(0)

            # chunk the passages into 256 token chunks by encoding and decoding
            passage_chunks = []
            for passage in passages:
                passage_encoded = tokenizer.encode(passage, truncation=True, max_length=128)
                passage_decoded = tokenizer.decode(passage_encoded)
                passage_chunks.append(passage_decoded)

            query = example.pop('query')

            prompt = ''
            for i, passage_chunk in enumerate(passage_chunks):
                psg = passage_chunk.replace('\n', ' ').strip()
                prompt += f"[{i+1}]: {psg}\n\n"
            prompt += f"Query: {query}\n\n" + instruction_following

            answer = target_index
            data = {
                "data_source": 'setwise-r1',
                "prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "ability": "ranking",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(target_index+1)
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=12)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=12)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)