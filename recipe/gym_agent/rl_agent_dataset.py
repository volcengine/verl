# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from omegaconf import ListConfig, DictConfig
import os
from typing import List, Union, Optional, Literal
import copy
import requests
import json
import pandas as pd
import textwrap

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

class RLAgentDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information

    The dataset is designed for RL training with the LLM as the agent. 
    It is built on top of the design in https://github.com/HMJiangGatech/verl_agent_env_examples.

    The dataset is organized as follows:
    ```json
    {
        "env_name": "verl_env/sokoban-v0",
        "seed": 0,
        "env_kwargs": null,
    }
    ```
    `env_name`: the name of the environment.
    `seed`: the seed for the environment.
    `env_kwargs`: the kwargs for the environment.
    They are used to initialize the environment, `initialize_env` in 
    `https://github.com/HMJiangGatech/verl_agent_env_examples/blob/master/src/verl_agent_env/app.py`.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.environment_endpoint = config.environment_endpoint
        # Test if the environment endpoint is valid
        try:
            response = requests.get(config.environment_endpoint)
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Invalid environment endpoint: {config.environment_endpoint}")

        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get('truncation', 'error')
        # TODO: implement the resume feature
        # whether to store the dataset in state_dict()
        # default not store
        # self.serialize_dataset = False
        self._download()
        self._read_files_and_initialize_env()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_initialize_env(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        raise NotImplementedError("Resume dataset state is not implemented for RLAgentDataset")
        # self.serialize_dataset = False if hasattr(self, 'original_data_files') else True
        # # resume dataframe if not it's serialized in data.pt
        # if not self.serialize_dataset:
        #     self._download(use_origin_parquet=True)  # download and resume from original parquet files
        #     self._read_files_and_tokenize()
        # else:
        #     print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        row_dict['index'] = torch.tensor(item, dtype=torch.int64)
        return row_dict

    def __getstate__(self):
        raise NotImplementedError("Serialize dataset is not implemented for RLAgentDataset")
        # if not self.serialize_dataset:
        #     state = self.__dict__.copy()

        #     if 'dataframe' in state:
        #         del state['dataframe']
        #     return state
        # return self.__dict__.copy()
