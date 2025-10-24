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

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import check_device_is_available
from verl.workers.sharding_manager.base import BaseShardingManager

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

        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
