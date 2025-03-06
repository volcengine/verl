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
Contains a resharding manager that binds weights from FSDP zero3 to XPerfGPT
"""
from .base import BaseShardingManager

from torch.distributed.device_mesh import DeviceMesh

from verl.utils.torch_functional import allgather_dict_tensors
from verl.protocol import all_gather_data_proto
from verl.utils.ulysses import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group
import numpy as np

import torch
import torch.distributed

from verl import DataProto


class FSDPUlyssesShardingManager(BaseShardingManager):
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.seed_offset = 12345

    def __enter__(self):
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh['sp'].get_group())
            # TODO: check how to set seed for each model

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
            # TODO: check how to set seed for each model

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from sp region
        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE
        In Ulysses, we need to make sure the same data is used across a SP group
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh['sp'].size()
            group = self.device_mesh['sp'].get_group()

            all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        Split the data to follow FSDP partition
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh['sp'].size()
            sp_rank = self.device_mesh['sp'].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]
        return data