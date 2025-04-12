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
Preprocess the ChartQA dataset to parquet format
"""

import os
import datasets
from datasets import Image as DatasetsImage
import argparse

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/chartqa')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'HuggingFaceM4/ChartQA'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = (
        'You first think about the reasoning process in the mind and then provides the user with the answer.'
        'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.'
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            image = example.pop('image')
            query = example.pop('query')
            label = example.pop('label')[0]
            human_or_machine = example.pop('human_or_machine')
            
            prompt = '<image>'+query + ' ' + instruction_following

            # Convert the image format to match geo3k format
            # This will create a Sequence(feature=Image(mode=None, decode=True, id=None))
            converted_image = image
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "images": [converted_image],
                "ability": "chart_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": label
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': label,
                    "question": query,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    # Cast the images field to the correct type to match geo3k format
    # This changes the schema from [{'bytes': binary, 'path': null}] to
    # Sequence(feature=Image(mode=None, decode=True, id=None), length=-1, id=None)
    # which ensures compatibility with other datasets like geo3k
    train_dataset = train_dataset.cast_column("images", datasets.Sequence(DatasetsImage()))
    test_dataset = test_dataset.cast_column("images", datasets.Sequence(DatasetsImage()))

    print(train_dataset.features)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 