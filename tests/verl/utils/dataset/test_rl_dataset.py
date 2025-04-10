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
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import OmegaConf


def get_gsm8k_data():
    # prepare test dataset
    url = "https://github.com/eric-haibin-lin/verl-data/raw/refs/heads/main/gsm8k/train.parquet"
    local_folder = os.path.expanduser('~/verl-data/gsm8k/')
    local_path = os.path.join(local_folder, 'train.parquet')
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer('deepseek-ai/deepseek-coder-1.3b-instruct')
    local_path = get_gsm8k_data()
    config = OmegaConf.create({
        "prompt_key": "prompt",
        "max_prompt_length": 256,
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": 2,
    })
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    data = dataset[0]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    print(f'type: type{output}')
    print(f'\n\noutput: {output}')
