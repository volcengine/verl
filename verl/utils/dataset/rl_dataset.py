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
from typing import List, Union, Optional
import copy
import datasets
from collections import defaultdict
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import ListConfig, DictConfig

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get('return_raw_chat', False)
        self.truncation = config.get('truncation', 'error')
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        self.mode = 'train' if any('/train/' in file or 'train.' in file for file in data_files) else 'other'
        self.sampling_ratios = config.get("sampling_ratios", None)
        if self.sampling_ratios is not None and len(self.sampling_ratios) != len(self.data_files):
            raise ValueError(f"Number of sampling_ratios ({len(self.sampling_ratios)}) must match number of data_files ({len(self.data_files)})")

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        dataset_sizes = []

        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
            dataset_sizes.append(len(dataframe))

        # If sampling_ratios is provided, apply them by sampling from each dataset
        if self.sampling_ratios is not None and self.mode == "train":
            combined_dataframes = []
            
            # Normalize ratios if they don't sum to 1
            ratios = self.sampling_ratios
            
            # Calculate target size for each dataset
            target_sizes = [int(dataset_sizes[ri] * ratio) for ri, ratio in enumerate(ratios)]
                
            # Sample from each dataset according to the target sizes
            for i, (dataframe, target_size) in enumerate(zip(dataframes, target_sizes)):
                if target_size == 0:
                    continue
                    
                if target_size < len(dataframe):
                    # Sample without replacement if we need fewer samples than available
                    sampled_indices = random.sample(range(len(dataframe)), target_size)
                    sampled_dataset = dataframe.select(sampled_indices)
                else:
                    # Use all samples if we need more or exactly what's available
                    sampled_dataset = dataframe
                    
                combined_dataframes.append(sampled_dataset)
                print(f'Dataset {i}: using {len(sampled_dataset)} samples (ratio: {ratios[i]:.2f}) out of {dataset_sizes[i]}')
                
            self.dataframe: datasets.Dataset = datasets.concatenate_datasets(combined_dataframes)
        else:
            # Concatenate all datasets
            self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
                               ) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_data_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            last_user_message: str = messages.pop(-1)['content']
            for i, content in enumerate(last_user_message.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)

        if self.image_key in row_dict:  # process multimodal data
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [process_image(image) for image in row_dict.pop(self.image_key)]
            model_inputs = self.processor(images, [raw_prompt], return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop('input_ids')
            attention_mask = model_inputs.pop('attention_mask')
            row_dict["multi_modal_data"] = {"image": images}
            row_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors='pt', add_special_tokens=False)
            input_ids = model_inputs.pop('input_ids')
            attention_mask = model_inputs.pop('attention_mask')

        input_ids, attention_mask = verl_F.postprocess_data(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            max_length=self.max_prompt_length,
                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                            left_pad=True,
                                                            truncation=self.truncation)

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict['raw_prompt_ids'] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state

        return self.__dict__.copy()
