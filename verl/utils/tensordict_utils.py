# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData


def assign_non_tensor_dict(tensor_dict: TensorDict, non_tensor_dict: dict):
    for key, val in non_tensor_dict.items():
        assign_non_tensor_data(tensor_dict=tensor_dict, key=key, val=val)
    return tensor_dict


def assign_non_tensor_data(tensor_dict: TensorDict, key, val):
    tensor_dict[key] = NonTensorData(val)


def get_tensordict(tensor_dict: dict[str, torch.Tensor | list], non_tensor_dict: dict = None) -> TensorDict:
    """

    Args:
        data_dict:
        meta_info:

    Returns:

    """
    if non_tensor_dict is None:
        non_tensor_dict = {}

    batch_size = None

    for key, val in tensor_dict.items():
        if isinstance(val, list):
            for v in val:
                assert not isinstance(v, torch.Tensor), "Passing a list makes the data NonTensorStack, which doesn't support torch.Tensor. Please convert to numpy first"

        assert isinstance(val, (torch.Tensor, list))

        if batch_size is None:
            batch_size = len(val)
        else:
            assert len(val) == batch_size

    for key, val in non_tensor_dict.items():
        assert key not in tensor_dict
        tensor_dict[key] = NonTensorData(val)

    return TensorDict(source=tensor_dict, batch_size=[batch_size])


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            if isinstance(tensor_dict2[key], torch.Tensor):
                assert tensor_dict1[key].equal(tensor_dict2[key]), (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )
            else:
                # non-tensor
                assert tensor_dict1[key] == tensor_dict2[key], (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )

    return tensor_dict1


def make_iterator(tensordict: TensorDict, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
    from torch.utils.data import DataLoader

    assert tensordict.batch_size[0] % mini_batch_size == 0, f"{tensordict.batch_size[0]} % {mini_batch_size} != 0"
    # we can directly create a dataloader from TensorDict
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    assert isinstance(dataloader_kwargs, dict)
    train_dataloader = DataLoader(
        dataset=tensordict, batch_size=mini_batch_size, collate_fn=lambda x: x, generator=generator, **dataloader_kwargs
    )

    def get_data():
        for _ in range(epochs):
            for d in train_dataloader:
                yield d

    return iter(get_data())
