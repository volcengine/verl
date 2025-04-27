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

from verl.utils import hf_tokenizer
from verl.utils.dataset.rm_dataset import RMDataset


def get_rm_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/verl-data/full_hh_rlhf/rm/")
    local_path = os.path.join(local_folder, "test.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rm_dataset():
    tokenizer = hf_tokenizer("facebook/opt-1.3b")
    local_path = get_rm_data()
    dataset = RMDataset(parquet_files=local_path, tokenizer=tokenizer, max_length=512)
    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode(data)
    assert len(output) > 1
    assert isinstance(output[0], str)
