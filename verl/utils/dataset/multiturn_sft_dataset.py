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

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor, PreTrainedTokenizer

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import compute_multimodal_position_ids, process_image
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

    def __init__(self, parquet_files: str | list[str], processor, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.images_key = multiturn_config.get("images_key", "images")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.processor: AutoProcessor = processor
        # for multi-modal processor, which always has a tokenizer for text to id
        if getattr(self.processor, "tokenizer", None) is not None:
            self.tokenizer: PreTrainedTokenizer = self.processor.tokenizer
        # for text models, processor is the same is tokenizer
        else:
            self.tokenizer = processor

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
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        if self.images_key in self.dataframe:
            self.images = self.dataframe[self.images_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.images = None

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

    def _process_message_tokens(
        self,
        encode_func,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_tokens = encode_func(
                messages[:start_idx], tools, enable_thinking=enable_thinking, add_generation_prompt=False
            )
            prev_applied_text = self.tokenizer.decode(prev_applied_tokens.input_ids[0])
            if is_assistant:
                prev_applied_text_w_generation_tokens = encode_func(
                    messages[:start_idx], tools, enable_thinking=enable_thinking, add_generation_prompt=True
                )
                prev_applied_text_w_generation_prompt = self.tokenizer.decode(
                    prev_applied_text_w_generation_tokens.input_ids[0]
                )

        else:
            prev_applied_text = ""

        cur_applied_tokens = encode_func(
            messages[:end_idx], tools, add_generation_prompt=False, enable_thinking=enable_thinking
        )
        cur_applied_text = self.tokenizer.decode(cur_applied_tokens.input_ids[0])

        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def encode_qwen25_vl(self, messages, tools, **kwargs):
        if "add_generation_prompt" in kwargs:
            add_generation_prompt = kwargs.pop("add_generation_prompt")
        else:
            add_generation_prompt = False
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        image_inputs, video_inputs = process_vision_info(messages)
        # enable_thinking and tools are invalid for qwen25 vl processor
        kwargs.pop("enable_thinking", None)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **kwargs,
            **self.apply_chat_template_kwargs,
        )
        return inputs

    def encode_pure_text(self, messages, tools, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            return_tensors="pt",
            **kwargs,
            **self.apply_chat_template_kwargs,
        )
        full_tokens = self.tokenizer([text], return_tensors="pt")
        return full_tokens

    def __getitem__(self, item):
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        images = self.images[item] if self.images is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        if images:
            for conv in messages:
                for content in conv["content"]:
                    if content["type"] == "image":
                        content["image"] = process_image(images[int(content["image"])])
            for conv in messages:
                for content in conv["content"]:
                    for k, v in content.items():
                        if v is None:
                            content.pop(k)
                            break

        if images:
            encode_func = self.encode_qwen25_vl
        else:
            encode_func = self.encode_pure_text

        # First, get the full conversation tokens
        try:
            full_tokens = encode_func(messages, tools)
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                f"{enable_thinking}"
            )
            raise

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    encode_func, messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    encode_func, messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    encode_func, messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

            # override loss mask with mask in the dataset to handle multi-turn conversation
            override_loss_mask = cur_messages.get("loss_mask", None)
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
                assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [override_loss_mask] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attention_mask)

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens.input_ids[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        if images:
            multi_modal_inputs = {
                "multi_modal_inputs_pixel_values": full_tokens.pixel_values,
                "multi_modal_inputs_image_grid_thw": full_tokens.image_grid_thw,
            }
        else:
            multi_modal_inputs = {}

        # encode prompt
        if messages[0]["role"] == "system":
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
        elif messages[0]["role"] == "user":
            assert messages[1]["role"] == "assistant"
        else:
            raise ValueError(f"Unknown role: {messages[0]['role']}")

        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
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

            position_ids = compute_multimodal_position_ids(
                processor=self.processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=full_tokens.get("image_grid_thw"),
                video_grid_thw=full_tokens.get("video_grid_thw"),
                second_per_grid_ts=full_tokens.get("second_per_grid_ts"),
            )

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]

            seq_len = len(input_ids)
            attention_mask = torch.ones(seq_len, dtype=torch.long)
            position_ids = compute_multimodal_position_ids(
                processor=self.processor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=full_tokens.get("image_grid_thw"),
                video_grid_thw=full_tokens.get("video_grid_thw"),
                second_per_grid_ts=full_tokens.get("second_per_grid_ts"),
            )

            # return nested tensor without padding
            result = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
        result.update(multi_modal_inputs)
        return result


if __name__ == "__main__":
    # the dataset loading script can be directly loaded
    parquet_files = "vermouth1992/mnist_multiturn_sft/data"
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    dataset = MultiTurnSFTDataset([parquet_files], tokenizer, {"pad_mode": "no_padding"})
    import ipdb

    ipdb.set_trace()
    print(dataset[1])
