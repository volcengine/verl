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

from tests.e2e.envs.digit_completion import DigitCompletion, generate_ground_truth_response
from torch.utils import data
import os

if __name__ == '__main__':
    simple_task = DigitCompletion(max_number=9, max_diff=9, max_num_in_response=9)
    all_prompts = simple_task.get_all_prompts()

    # 21 * 6 * 4
    train_data, test_data = data.random_split(all_prompts, lengths=[0.8, 0.2])
    train_data = list(train_data)
    test_data = list(test_data)

    train_data = [[{'role': 'user', 'content': str(item)}] \
                     for item in train_data]
    test_data = [[{'role': 'user', 'content': str(item)}] \
                     for item in test_data]

    print(f'Size of train: {len(train_data)}, size of test: {len(test_data)}')

    train_data = {'prompt': train_data}
    test_data = {'prompt': test_data}

    model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    import pandas as pd

    train_data_frame = pd.DataFrame(train_data)
    test_data_frame = pd.DataFrame(test_data)

    train_data_frame.to_parquet(os.path.join(model_folder, 'train.parquet'))
    test_data_frame.to_parquet(os.path.join(model_folder, 'test.parquet'))
