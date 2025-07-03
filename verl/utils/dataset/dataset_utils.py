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

from collections import defaultdict
from typing import List

import numpy as np
import torch

COLLATE_FN_MANAGER_REGISTRY = {}


def _pad_for_batching(
    pixel_values: List[torch.Tensor],
    image_sizes: List[List[int]],
):
    """
    Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.
    Args:
        pixel_values (`List[torch.Tensor]`):
            An array of pixel values of each images of shape (`batch_size`, `channels`, `height`, `width`)
        image_sizes (`List[List[int]]`):
            A list of sizes for each image in `pixel_values` in (height, width) format.
    Returns:
        List[`torch.Tensor`]: The padded images.
    """
    max_shape = (max([size[0] for size in image_sizes]), max([size[1] for size in image_sizes]))
    pixel_values = [torch.nn.functional.pad(image, pad=(0, max_shape[1] - size[1], 0, max_shape[0] - size[0])).unsqueeze(0) for image, size in zip(pixel_values, image_sizes)]
    return pixel_values


def register_collate_fn(name):
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls):
        if name in COLLATE_FN_MANAGER_REGISTRY and COLLATE_FN_MANAGER_REGISTRY[name] != cls:
            raise ValueError(f"Collate function manager {name} has already been registered: {COLLATE_FN_MANAGER_REGISTRY[name]} vs {cls}")
        COLLATE_FN_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_collate_fn_manager_cls(name):
    """Get the collate function manager class with a given name.

    Args:
        name: `(str)`
            The name of the collate function manager.

    Returns:
        `(type)`: The collate function manager class.
    """
    if name not in COLLATE_FN_MANAGER_REGISTRY:
        default_collate_fn = COLLATE_FN_MANAGER_REGISTRY.get("default", None)
        if default_collate_fn is None:
            raise ValueError(f"Unknown collate function manager: {name}")
        return default_collate_fn

    return COLLATE_FN_MANAGER_REGISTRY[name]


@register_collate_fn("default")
def collate_fn(data_list: List[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
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


@register_collate_fn("PixtralProcessor")
def collate_fn_for_pixtral(data_list: List[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
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
        if key == "multi_modal_inputs":
            val = [ele for ele in val if ele]
            if not val:
                continue
            pixel_values = [v["pixel_values"][0] for v in val]
            image_sizes = [v["image_sizes"][0] for v in val]
            pixel_values = _pad_for_batching(pixel_values, image_sizes)
            for v, pixel_value in zip(val, pixel_values):
                v["pixel_values"] = pixel_value
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}
