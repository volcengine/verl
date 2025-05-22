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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.vision_utils import process_image, process_video

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class VLMSFTDataset(SFTDataset):
    """SFT Dataset for Vision-Language Models"""

    def __init__(
        self, 
        parquet_files: Union[str, List[str]], 
        tokenizer, 
        config, 
        processor=None
    ):
        self.processor = processor
        self.image_key = config.get("image_key", "images")
        super().__init__(parquet_files, tokenizer, config)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        processor = self.processor

        prompt = self.prompts[item]
        response = self.responses[item]

        # Process images if available
        multi_modal_inputs = {}
        images = None
        
        if self.image_key in self.dataframe.columns:
            images = self.dataframe[self.image_key].iloc[item]
            if images is not None:
                if not isinstance(images, list):
                    images = [images]
                # Process images using the utility function
                images = [process_image(image) for image in images]

        # Apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]
        
        # Process with processor if available
        if processor is not None and images:
            # Process text and images together using the processor
            raw_prompt = processor.tokenizer.apply_chat_template(
                prompt_chat, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process inputs with the VLM processor
            model_inputs = processor(
                text=[raw_prompt], 
                images=images, 
                return_tensors="pt"
            )
            
            # Extract the basic inputs
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            
            # Store the remaining multimodal inputs
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    multi_modal_inputs[key] = value[0]  # Remove batch dimension
                else:
                    multi_modal_inputs[key] = value
            
            # For Qwen models, we need to handle position IDs specially
            if hasattr(processor, "image_processor") and processor.image_processor.__class__.__name__ in [
                "Qwen2VLImageProcessor", 
                "Qwen2_5VLImageProcessor"
            ]:
                # Import the necessary function
                from verl.models.transformers.qwen2_vl import get_rope_index
                
                # Get position IDs for the VLM
                position_ids, _ = get_rope_index(
                    input_ids=input_ids.unsqueeze(0),
                    image_grid_thw=multi_modal_inputs.get("image_grid_thw", None),
                    video_grid_thw=None,
                    attention_mask=attention_mask.unsqueeze(0)
                )
                position_ids = position_ids[0]  # Remove batch dimension
            else:
                # Default position IDs calculation
                position_ids = compute_position_id_with_mask(attention_mask)
        else:
            # Standard text-only processing
            return super().__getitem__(item)
        
        # Process response
        response_chat_str = response + tokenizer.eos_token
        response_ids_output = tokenizer(
            response_chat_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        
        # Combine prompt and response
        prompt_length = input_ids.shape[0]
        response_length = response_ids.shape[0]
        
        input_ids = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Handle padding and truncation
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(
                size=(self.max_length - sequence_length,), 
                dtype=input_ids.dtype
            ) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(
                size=(self.max_length - sequence_length,), 
                dtype=attention_mask.dtype
            )
            
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
        
        # Create loss mask
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        
        # Add multimodal inputs
        if multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs
        
        return result

        
