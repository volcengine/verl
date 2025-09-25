# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Huawei Ltd. and/or its affiliates
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

from enum import Enum

import ray
import torch
from tensordict import TensorDict


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class TransferQueueRole(ExplicitEnum):
    CONTROLLER = "TransferQueueController"
    STORAGE = "TransferQueueStorage"
    CLIENT = "TransferQueueClient"


# production_status enum: 0: not produced, 1: ready for consume, 2: consumed
class ProductionStatus(ExplicitEnum):
    NOT_PRODUCED = 0
    READY_FOR_CONSUME = 1
    CONSUMED = 2


def get_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1):
    """
    Create a placement group with SPREAD strategy for Ray actors.

    Args:
        num_ray_actors (int): Number of Ray actors to create.
        num_cpus_per_actor (int): Number of CPUs to allocate per actor.

    Returns:
        placement_group: The created placement group.
    """
    bundle = {"CPU": num_cpus_per_actor}
    placement_group = ray.util.placement_group([bundle for _ in range(num_ray_actors)], strategy="SPREAD")
    ray.get(placement_group.ready())
    return placement_group


def random_sampler(
    ready_for_consume_idx: list[int],
    batch_size: int,
    get_n_samples: bool,
    n_samples_per_prompt: int,
) -> list[int]:
    """
    random sampling batch_size samples from global indexes ready_for_consume_idx
    input example:
        if get_n_samples: (group_num=3, group_size=4)
            ready_for_consume_idx could look like: [0, 1, 2, 3,   8, 9, 10, 11,   16, 17, 18, 19]
        else:
            ready_for_consume_idx could look like: [2, 5, 6]
    """
    if get_n_samples:
        assert len(ready_for_consume_idx) % n_samples_per_prompt == 0
        assert batch_size % n_samples_per_prompt == 0
        batch_size_n_samples = batch_size // n_samples_per_prompt

        group_ready_for_consume_idx = torch.tensor(ready_for_consume_idx, dtype=torch.int).view(
            -1, n_samples_per_prompt
        )

        weights = torch.ones(group_ready_for_consume_idx.size(0))
        sampled_indexes_idx = torch.multinomial(weights, batch_size_n_samples, replacement=False).tolist()
        sampled_indexes = group_ready_for_consume_idx[sampled_indexes_idx].flatten().tolist()
    else:
        weights = torch.ones(len(ready_for_consume_idx))
        sampled_indexes_idx = torch.multinomial(weights, batch_size, replacement=False).tolist()
        sampled_indexes = [int(ready_for_consume_idx[i]) for i in sampled_indexes_idx]
    return sampled_indexes


def extract_field_info(tensor_dict: TensorDict) -> dict:
    """
    Extract field names, dtypes, and shapes from a TensorDict.
    Assumes all tensors in the same field have the same dtype and shape (excluding batch dimension).
    Returns a dictionary with keys: 'names', 'dtypes', 'shapes'.
    """
    field_info: dict[str, list] = {"names": [], "dtypes": [], "shapes": []}
    for key, value in tensor_dict.items():
        field_info["names"].append(key)

        # TODO: support nested tensors & non tensors
        # field_info["dtypes"].append(value.dtype)
        # field_info["shapes"].append(value.shape[1:])  # exclude batch dimension
    return field_info
