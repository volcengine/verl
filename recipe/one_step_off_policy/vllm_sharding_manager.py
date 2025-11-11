# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import os
from typing import Any

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device, get_device_id
from verl.utils.torch_functional import check_device_is_available
from verl.workers.sharding_manager.base import BaseShardingManager
import numpy as np
import torch
from tensordict import TensorDict
from verl.utils.torch_functional import allgather_dict_tensors

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(self, inference_engine, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh
        self.inference_engine = inference_engine
        inference_engine.wake_up()
        assert device_mesh is not None
        assert inference_engine is not None
        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()
        self.timing = {}
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def __enter__(self):
        get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        self.gen_random_states = get_torch_device().get_rng_state()
        self.inference_engine.reset_prefix_cache()

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # Use vLLM's coordinator to access device and CPU groups explicitly
        tp = vllm_ps.get_tensor_model_parallel_group()
        dev_group = tp.device_group
        cpu_group = tp.cpu_group

        # Sanity: ensure same tensor keys across ranks to avoid NCCL mismatches
        local_keys = sorted(list(data.batch.keys())) if data.batch is not None else []
        keys_lists = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(keys_lists, local_keys, group=cpu_group)
        union_keys = sorted({k for ks in keys_lists for k in (ks or [])})
        if set(local_keys) != set(union_keys):
            raise RuntimeError(
                f"Inconsistent DataProto.batch keys across TP ranks. local={local_keys}, union={union_keys}"
            )

        # 1) Ensure all ranks have the same batch size via right padding
        local_bsz = data.batch.batch_size[0] if data.batch is not None else 0
        world_size = torch.distributed.get_world_size(group=dev_group)
        bsz_list: list[int] = [0 for _ in range(world_size)]
        torch.distributed.all_gather_object(bsz_list, int(local_bsz), group=cpu_group)
        total_bsz = int(sum(bsz_list))
        max_bsz = int(max(bsz_list))

        if data.batch is not None and local_bsz < max_bsz:
            padded = {}
            for k, t in data.batch.items():
                if t.dim() == 0:
                    pad = torch.zeros((max_bsz,), dtype=t.dtype, device=t.device)
                    pad[:local_bsz] = t.expand(local_bsz)
                else:
                    pad = torch.zeros((max_bsz,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
                    pad[:local_bsz] = t
                padded[k] = pad
            data.batch = TensorDict(padded, batch_size=[max_bsz])

        # 2) Move to device for NCCL tensor gathers
        prev_device = data.batch.device if data.batch is not None else None
        data = data.to(get_device_id())
        data.batch = allgather_dict_tensors(data.batch.contiguous(), size=world_size, group=dev_group, dim=0)
        if total_bsz < data.batch.batch_size[0]:
            data.batch = data.batch[:total_bsz]
        if prev_device is not None:
            data = data.to(prev_device)

        # 3) Gather non-tensor payloads over CPU group
        all_non_tensor: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore
        torch.distributed.all_gather_object(all_non_tensor, data.non_tensor_batch, group=cpu_group)
        if data.non_tensor_batch is not None and len(data.non_tensor_batch) > 0:
            data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor]) for k in data.non_tensor_batch}

        return data

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
