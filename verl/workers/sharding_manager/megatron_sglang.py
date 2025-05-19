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
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

import logging
import os

import torch
from torch import nn

from verl.utils.debug import log_gpu_memory_usage

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))
"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import torch.distributed
from sglang.srt.entrypoints.verl_engine import VerlEngine
from torch.distributed import new_group

from verl.utils.debug import GPUMemoryLogger
from verl.utils.megatron_utils import per_tensor_generator

from .base import BaseShardingManager

_MICRO_DATA_PARALLEL_GROUP = None


class MegatronSGLangShardingManager(BaseShardingManager):

    def __init__(self, actor_module: nn.ModuleList, inference_engine: VerlEngine, model_config, layer_name_mapping, weight_converter):
        from megatron.core import parallel_state as mpu
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        global _MICRO_DATA_PARALLEL_GROUP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        self.infer_tp_size = self.inference_engine._tp_size
        self.train_tp_size = mpu.get_tensor_model_parallel_world_size()
        self.need_tp_reshard = self.infer_tp_size == self.train_tp_size

        assert self.infer_tp_size <= self.train_tp_size, \
            'Not implemented for infer_tp > train_tp'
        assert self.train_tp_size % self.infer_tp_size == 0

        micro_dp_size = self.train_tp_size // self.infer_tp_size
        num_micro_dp_groups = world_size // micro_dp_size
        assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
        for i in range(num_micro_dp_groups):
            ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
            group = new_group(ranks=ranks)
            if rank in ranks:
                _MICRO_DATA_PARALLEL_GROUP = group

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        per_tensor_param = per_tensor_generator(self.actor_module, self.model_config, self.weight_converter, self.layer_name_mapping)
        self.inference_engine.resume_memory_occupation()
        self.inference_engine.update_weights_from_tensor(per_tensor_param, load_format=None)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before SGLang offload in sharding manager', logger=logger)
        self.inference_engine.release_memory_occupation()
        log_gpu_memory_usage('After SGLang offload in sharding manager', logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        torch.cuda.empty_cache()
