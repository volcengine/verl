# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

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

import numpy
import numpy as np
import pandas
import pandas as pd
import torch
from omegaconf import ListConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length, postprocess_data


def convert_nested_value_to_list_recursive_if_not_none(data_item):
    if isinstance(data_item, pandas.core.series.Series | numpy.ndarray) and len(data_item) == 1:
        return convert_nested_value_to_list_recursive_if_not_none(data_item[0])
    elif isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive_if_not_none(v) for k, v in data_item.items() if v is not None}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive_if_not_none(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive_if_not_none(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], processor, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "left_right"], (
            f"Expect pad_mode to be 'right' or 'left_right'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # for left right paddding to be consistent with RL
        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.max_response_length = config.get("max_response_length", 512)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.processor: AutoProcessor = processor

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = (
            self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive_if_not_none).tolist()
        )

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = (
                self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive_if_not_none).tolist()
            )
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def __len__(self):
        return len(self.messages)

    def _get_pad_id(self):
        if hasattr(self.processor, "tokenizer"):
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.processor
        if getattr(tokenizer, "pad_token_id", None) is not None:
            return tokenizer.pad_token_id
        # for qwen2.5 vl
        if getattr(tokenizer, "pad_token_type_id", None) is not None:
            return tokenizer.pad_token_type_id
        return 0

    def __getitem__(self, item):
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        full_text = self.processor.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            return_tensors="pt",
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        if getattr(self.processor, "image_token", None):
            images, videos = process_vision_info(messages)
            tokens = self.processor(text=[full_text], images=images, videos=videos, padding=False)
        else:
            tokens = self.processor(text=[full_text], padding=False)
        input_ids = tokens.input_ids[0]
        attention_mask = tokens.attention_mask[0]
        pixel_values = getattr(tokens, "pixel_values", None)
        image_grid_thw = getattr(tokens, "image_grid_thw", None)

        loss_mask = [0] * len(input_ids)
        empty_with_gen_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": "123"}], add_generation_prompt=True, tokenize=False
        )
        empty_without_gen_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": "123"}], add_generation_prompt=False, tokenize=False
        )
        gen_prompt = empty_with_gen_prompt[len(empty_without_gen_prompt) :]
        if hasattr(self.processor, "tokenizer"):
            gen_tokens = self.processor.tokenizer.encode(gen_prompt)
            end_tokens = [self.processor.tokenizer.encode(empty_without_gen_prompt.strip())[-1]]
        else:
            gen_tokens = self.processor.encode(gen_prompt)
            end_tokens = [self.processor.encode(empty_without_gen_prompt.strip())[-1]]
        start_indexes = []
        end_indexes = []
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(gen_tokens)] == gen_tokens:
                start_indexes.append(i + len(gen_tokens))
                i += len(gen_tokens)
                while i < len(input_ids):
                    if input_ids[i : i + len(end_tokens)] == end_tokens:
                        end_indexes.append(i + len(end_tokens))
                        break
                    i += 1
            i += 1
        assert len(start_indexes) == len(end_indexes)
        for start, end in zip(start_indexes, end_indexes, strict=False):
            assert end > start
            loss_mask[start:end] = [1] * (end - start)

        input_ids, loss_mask, attention_mask = (
            torch.tensor(input_ids),
            torch.tensor(loss_mask),
            torch.tensor(attention_mask),
        )

        # encode prompt
        if messages[0]["role"] == "system":
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            prompt_message_length = 2
        elif messages[0]["role"] == "user":
            assert messages[1]["role"] == "assistant"
            prompt_message_length = 1
        else:
            raise ValueError(f"Unknown role: {messages[0]['role']}")

        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == "right":
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self._get_pad_id()
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

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

            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask,
                )  # (3, seq_length)
                valid_mask = attention_mask.bool()
                text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (1, 4, seq_length)
            else:
                # Create position IDs
                position_ids = torch.arange(len(input_ids), dtype=torch.long)
                # Zero out position IDs for padding
                position_ids = position_ids * attention_mask

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": input_ids,
                "loss_mask": loss_mask,
                "response_mask": loss_mask,
            }
        elif self.pad_mode == "left_right":
            assert self.truncation == "error", "Only support error truncation for left_right pad mode"
            prompt_str = self.processor.apply_chat_template(
                messages[:prompt_message_length],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
            prompt_ids = self.processor.encode(prompt_str, add_special_tokens=False)
            prompt_length = len(prompt_ids)
            prompt_ids = input_ids[:prompt_length].unsqueeze(0)
            prompt_attention_mask = attention_mask[:prompt_length].unsqueeze(0)
            prompt_loss_mask = loss_mask[:prompt_length].unsqueeze(0)
            response_ids = input_ids[prompt_length:].unsqueeze(0)
            response_attention_mask = attention_mask[prompt_length:].unsqueeze(0)
            response_loss_mask = loss_mask[prompt_length:].unsqueeze(0)

            assert prompt_loss_mask.sum().item() == 0

            prompt_ids, prompt_attention_mask = postprocess_data(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            response_ids, response_attention_mask = postprocess_data(
                input_ids=response_ids,
                attention_mask=response_attention_mask,
                max_length=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.truncation,
            )
            response_loss_mask = pad_sequence_to_length(
                response_loss_mask, max_seq_len=self.max_response_length, pad_token_id=0, left_pad=False
            )

            prompt_ids = prompt_ids[0]
            prompt_attention_mask = prompt_attention_mask[0]
            response_ids = response_ids[0]
            response_attention_mask = response_attention_mask[0]
            response_loss_mask = response_loss_mask[0]

            assert response_attention_mask[0].item() == 1
            assert response_loss_mask[0].item() == 1

            input_ids = torch.cat((prompt_ids, response_ids), dim=0)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=0)
            position_ids = compute_position_id_with_mask(attention_mask)

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
            }
        else:
            raise NotImplementedError("pad_mode only support right or left-right mode!")
        if pixel_values is not None:
            result["multi_modal_inputs"] = {}
            result["multi_modal_inputs"]["pixel_values"] = torch.tensor(pixel_values)
            result["multi_modal_inputs"]["image_grid_thw"] = torch.tensor(image_grid_thw)
        return result
