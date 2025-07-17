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

META_INFO_KEY = "meta_info"


def get_tensordict(tensor_dict: dict[str, torch.Tensor | list], meta_info: dict = None) -> TensorDict:
    """

    Args:
        data_dict:
        meta_info:

    Returns:

    """
    if meta_info is None:
        meta_info = {}

    batch_size = None

    for key, val in tensor_dict.items():
        assert key != META_INFO_KEY
        assert isinstance(val, (torch.Tensor, list))

        if batch_size is None:
            batch_size = len(val)
        else:
            assert len(val) == batch_size

    meta_info = NonTensorData(meta_info)
    tensor_dict.update({META_INFO_KEY: meta_info})
    return TensorDict(source=tensor_dict, batch_size=[batch_size])


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
        dataset=tensordict, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
    )

    def get_data():
        for _ in range(epochs):
            for d in train_dataloader:
                d.meta_info = self.meta_info
                yield d

    return iter(get_data())
