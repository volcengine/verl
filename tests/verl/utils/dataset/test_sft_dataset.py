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

from transformers import AutoTokenizer
from verl.utils import set_pad_token_id
from verl.utils.dataset.sft_dataset import SFTDataset


def get_gsm8k_data():
    # prepare test dataset
    url = "https://github.com/eric-haibin-lin/verl-data/raw/refs/heads/main/gsm8k/train.parquet"
    local_folder = os.path.expanduser('/opt/tiger/verl-data/gsm8k/')
    local_path = os.path.join(local_folder, 'train.parquet')
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_sft_dataset():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    set_pad_token_id(tokenizer)
    local_path = get_gsm8k_data()
    dataset = SFTDataset(parquet_files=local_path,
                         tokenizer=tokenizer,
                         prompt_key='prompt',
                         prompt_dict_keys=['content'],
                         response_key='extra_info',
                         response_dict_keys=['answer'],
                         max_length=512)

    data = dataset[0]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    assert len(output) > 1
    assert type(output) == str