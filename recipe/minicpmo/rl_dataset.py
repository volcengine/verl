# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import math
import os
import re
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.vision_utils import process_image
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
    )


def build_image_bound(input_ids, tokenizer, new_schema=True, logger=None):
    if new_schema:
        start_cond = (input_ids == tokenizer.im_start_id) | (input_ids == tokenizer.slice_start_id)
        end_cond = (input_ids == tokenizer.im_end_id) | (input_ids == tokenizer.slice_end_id)
    else:
        start_cond = input_ids == tokenizer.im_start_id
        end_cond = input_ids == tokenizer.im_end_id
    image_start_tokens = torch.where(start_cond)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(end_cond)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        logger.error("image start token != image end tokens")
        raise Exception("image start token != image end tokens")
    if len(image_start_tokens) > 0:
        image_bound = torch.hstack([image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)])
    else:
        image_bound = []
    return image_bound


def preprocess(
    images_dict,
    conversations,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
    max_length=2048,
    truncation="error",
    logger=None,
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert conversations[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    new_schema = False
    use_image_id = False
    if llm_type == "qwen":
        new_schema = True
        use_image_id = True
    image_placeholder_dict = {}
    images = []
    image_id_cnt = 0
    for img_name, image in images_dict.items():
        if slice_config:
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder
            if len(patches) > 0:
                for i in range(len(patches)):
                    for j in range(len(patches[0])):
                        images.append(patches[i][j])
                if use_image_id:
                    image_placeholder = (
                        f"{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}" + image_placeholder
                    )
                    image_id_cnt += 1
                image_placeholder += get_grid_placeholder(tokenizer, best_grid, query_nums, new_schema=new_schema)
            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            if use_image_id:
                image_placeholder = f"{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}" + image_placeholder
                image_id_cnt += 1
            else:
                image_placeholder = default_image_placeholder
            image_placeholder_dict[img_name] = image_placeholder

    images = [transform(i) for i in images]

    if len(images_dict) == 1 and "<image>" in images_dict:
        if "<image>" in conversations[0]["content"]:
            conversations[0]["content"] = conversations[0]["content"].replace("<image>", image_placeholder)
        else:
            conversations[0]["content"] = image_placeholder + "\n" + conversations[0]["content"]
    else:
        pattern = r"<image_\d+>"
        new_conversations = []
        for conversation in conversations:
            content = conversation["content"]
            parts = re.split(f"({pattern})", content)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                if re.match(pattern, part):
                    if part in image_placeholder_dict:
                        parts[i] = image_placeholder_dict[part]
                    else:
                        raise Exception(f"not found {part} in image dict")
            conversation["content"] = "\n".join(parts)
            new_conversations.append(conversation)
        conversations = new_conversations

    # TODO change role in conversation for different llm
    prompt_with_chat_template = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)

    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=prompt_with_chat_template,
        tokenizer=tokenizer,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    image_bound = build_image_bound(input_ids[0], tokenizer, new_schema, logger)

    input_dict = {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
        "image_bound": image_bound,
    }

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size, allow_upscale=True)

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(original_size, grid, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
    else:
        image_placeholder = tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    if new_schema:
        slice_placeholder = "\n".join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size))

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(image_tensor.size(0), patch_size, -1)
    return patches


def init_minicpmo_config(processor, config):
    """Initialize MiniCPM-o specific configuration"""
    minicpmo_config = {
        "transform": build_transform(),
        "patch_size": config.get("patch_size", 14),
        "query_nums": config.get("query_nums", 64),
        "slice_config": config.get(
            "slice_config", {"max_slice_nums": 9, "patch_size": config.get("patch_size", 14), "scale_resolution": 448}
        ),
        "llm_type": config.get("llm_type", "qwen"),
        "batch_vision": config.get("batch_vision", True),
    }
    return minicpmo_config


def process_minicpmo_data(
    row_dict, messages, tokenizer, minicpmo_config, image_key, max_prompt_length, truncation, logger
):
    """Process data for MiniCPM-o model"""
    if len(row_dict[image_key]) == 1:
        multi_modal_data = {}
        image = process_image(row_dict.pop(image_key)[0])
        multi_modal_data["image"] = [image]
        images_dict = {"<image>": image}
    else:
        raise NotImplementedError

    model_inputs = preprocess(
        images_dict,
        messages,
        tokenizer,
        minicpmo_config["transform"],
        query_nums=minicpmo_config["query_nums"],
        slice_config=minicpmo_config["slice_config"],
        llm_type=minicpmo_config["llm_type"],
        patch_size=minicpmo_config["patch_size"],
        batch_vision=minicpmo_config["batch_vision"],
        max_length=max_prompt_length,
        truncation=truncation,
        logger=logger,
    )

    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    raw_prompt = raw_prompt.replace("<image>", "(<image>./</image>)")

    return model_inputs, multi_modal_data, raw_prompt


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.minicpmo_config = init_minicpmo_config(self.processor, config)
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        return example.pop(self.prompt_key)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            model_inputs, multi_modal_data, raw_prompt = process_minicpmo_data(
                row_dict,
                messages,
                self.tokenizer,
                self.minicpmo_config,
                self.image_key,
                self.max_prompt_length,
                self.truncation,
                logger,
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            position_ids = model_inputs.pop("position_ids")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
