# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
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

import datasets
import torch
from torch.utils.data import Dataset


class AgentDataset(Dataset):
    def __init__(self, data_files, *args, **kwargs):
        super().__init__()

        # Load dataset from parquet files
        if not isinstance(data_files, list):
            data_files = [data_files]

        dataframes = []
        for parquet_file in data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # Workaround for data proto. At least one tensor is needed.
        row_dict["fake_ids"] = torch.ones(1, dtype=torch.int)
        return row_dict
