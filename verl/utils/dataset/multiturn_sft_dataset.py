# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()
        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            print(f"Messages: {messages}")
            print(f"Tools: {tools}")
            print(f"Enable thinking: {enable_thinking}")
            raise e
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                prev_applied_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
                cur_applied_text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
                
                # Get token positions for the assistant response
                prev_tokens = tokenizer.encode(prev_applied_text, add_special_tokens=False)
                cur_tokens = tokenizer.encode(cur_applied_text, add_special_tokens=False)
                start_pos = len(prev_tokens)
                end_pos = len(cur_tokens)
                
                # Set loss mask for assistant response
                loss_mask[start_pos:end_pos] = 1
                i += 1
            elif cur_messages["role"] == "tool":
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                prev_applied_text = tokenizer.apply_chat_template(messages[st:ed], tokenize=False, add_generation_prompt=False)
                cur_applied_text = tokenizer.apply_chat_template(messages[st:ed+1], tokenize=False, add_generation_prompt=False)
                
                i = ed
            elif cur_messages["role"] == "user":
                prev_applied_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
                cur_applied_text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
                i += 1
            elif cur_messages["role"] == "system":
                prev_applied_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
                cur_applied_text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
