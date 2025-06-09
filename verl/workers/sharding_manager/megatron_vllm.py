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

import inspect
import logging
import os

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from torch import nn

from verl import DataProto
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.debug.performance import _timer
from verl.utils.megatron_utils import (
    per_tensor_generator,
)
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank 
   to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


class MegatronVLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: LLM,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        # initialize groups for vllm inference
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.infer_tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.infer_tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.infer_tp_group = vllm_ps.get_tensor_model_parallel_group()
        if vllm_version not in ("0.5.4", "0.6.3"):
            self.infer_tp_group = self.infer_tp_group.device_group
        self.train_tp_size = mpu.get_tensor_model_parallel_world_size()
        self.train_tp_rank = mpu.get_tensor_model_parallel_rank()
        self.train_tp_group = mpu.get_tensor_model_parallel_group()
        self.train_ep_size = mpu.get_expert_model_parallel_world_size()
        self.train_ep_rank = mpu.get_expert_model_parallel_rank()
        self.train_ep_group = mpu.get_expert_model_parallel_group()
        self.train_etp_size = mpu.get_expert_tensor_parallel_world_size()
        self.train_etp_rank = mpu.get_expert_tensor_parallel_rank()
        self.train_etp_group = mpu.get_expert_tensor_parallel_group()
        self.need_tp_reshard = self.train_tp_size != self.infer_tp_size
        self.train_tp_larger = self.train_tp_size > self.infer_tp_size

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        self.timing = {}
        with _timer("reshard", self.timing):
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                per_tensor_param = per_tensor_generator(self.actor_module, self.model_config, self.weight_converter, self.transformer_config, self.layer_name_mapping, convert_qkv_gate_up_by_simple_split=False)
                self.inference_engine.sync_model_weights(per_tensor_param, load_format="megatron")
            else:
                # > 0.7.2
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["weights"])
                else:
                    self.inference_engine.wake_up()
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
                model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
                patch_vllm_moe_model_weight_loader(model)
                loaded_params = model.load_weights(per_tensor_param)
                info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
                logger.info(info)

            # (vermouth1992) We move wake up kv cache after we release model weights. Need refactor to make API cleaner
            # if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            #     self.inference_engine.wake_up(tags=["kv_cache"])

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        for model in self.actor_module:
            model.train()

        torch.cuda.empty_cache()

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.infer_tp_group)
        return data

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.infer_tp_rank]
