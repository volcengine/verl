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

import importlib
import logging
import os
import torch
import torch.distributed as dist
from torch import nn

from verl.utils.model import normalize_model_name
from verl.utils.megatron_utils import broadcast_from_megatron_pp, broadcast_str_from_megatron_pp

from verl.utils.megatron_utils import get_model, unwrap_model
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.megatron_utils import convert_megatron_model_to_transformers_model

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

from .base import BaseShardingManager

import torch
from torch import nn
import torch.distributed
from torch.distributed import new_group
from torch.distributed._tensor import DTensor
from typing import Dict, Iterable, Union, Tuple

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from sglang.srt.entrypoints.verl_engine import VerlEngine

import verl.utils.megatron.tensor_parallel as tp_utils
from verl.utils.model import normalize_pp_vpp_params

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
        self.train_tp_rank = mpu.get_tensor_model_parallel_rank()
        self.train_tp_group = mpu.get_tensor_model_parallel_group()
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

    def per_tensor_generator(self, convert_qkv_gate_up_by_simple_split=True):
        from megatron.core import parallel_state as mpu

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        vpp_size = len(self.actor_module)

        all_gather_group = (self.train_tp_group)

        all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)

        def tensor_generator():
            for scan_vpp_idx in range(vpp_size):
                yield from self.actor_module[scan_vpp_idx].named_parameters()

        # we need first make all rank get full model information
        meta_info = []
        for scan_vpp_idx in range(vpp_size):
            for idx, (name, _) in enumerate(self.actor_module[scan_vpp_idx].named_parameters()):
                meta_info.append((pp_rank, scan_vpp_idx, idx, name))

        obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
        torch.distributed.all_gather_object(
            object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group()
        )
        layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

        gen_func = tensor_generator()

        # lazy load tensor for full model
        for cur_pp_rank, scan_vpp_idx, idx, name in layer_list_meta:
            if cur_pp_rank == pp_rank:
                try:
                    cur_name, cur_tensor = next(gen_func)
                except StopIteration:
                    cur_name, cur_tensor = None, None
                cur_name = normalize_model_name(
                    name, cur_pp_rank, scan_vpp_idx, pp_size, vpp_size, self.model_config.num_hidden_layers
                )
            else:
                cur_tensor, cur_name = None, None

            # pp broadcast model tensor and name
            cur_name = broadcast_str_from_megatron_pp(cur_name)
            broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

            # (xya): this is a hack to fix the name of the parameters
            while cur_name.startswith("module."):
                cur_name = cur_name[len("module.") :]

            # tp all gather
            if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
                # allocate a new tensor with proper size
                if all_gather_group_size <= 1:
                    infer_params = [broad_pp_tensor]
                else:
                    infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                    torch.distributed.all_gather(
                        infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group()
                    )
                infer_params = self.default_tp_concat_fn(
                    cur_name, broad_pp_tensor, infer_params, self.model_config, convert_qkv_gate_up_by_simple_split
                )
            else:
                infer_params = broad_pp_tensor


            if not isinstance(infer_params, list):
                infer_params = [infer_params]
            converted_names, converted_params = self.weight_converter.convert_param(cur_name, infer_params)

            yield from zip(converted_names, converted_params)

    def default_tp_concat_fn(self, name, param, infer_params, model_config, convert_qkv_gate_up_by_simple_split=False):
        """
        name: name of the parameter
        param: training parameters
        infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered from micro_dp_group
        model_config: huggingface model_config
        TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
        definition so that it is model-agnostic. If the model doesn't implement this function, 
        we can throw an error to force user disable TP HybridEngine.
        """
        from megatron.core import mpu

        if self.layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            q_lst = []
            k_lst = []
            v_lst = []
            assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
            num_q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
            assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0
            kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
            for infer_param in infer_params:
                num_query_groups_per_partition = model_config.num_key_value_heads // mpu.get_tensor_model_parallel_world_size(
                )
                for chunk in infer_param.chunk(num_query_groups_per_partition):
                    split_size = [
                        kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition
                    ]
                    q, k, v = chunk.split(split_size)
                    q_lst.append(q)
                    k_lst.append(k)
                    v_lst.append(v)
            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)
            if not convert_qkv_gate_up_by_simple_split:
                infer_params = torch.cat((q, k, v), dim=0)
            else:
                infer_params = [q, k, v]

        elif self.layer_name_mapping.get("gate_proj_layer_name") in name:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for infer_param in infer_params:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            if not convert_qkv_gate_up_by_simple_split:
                infer_params = torch.cat((gate, up), dim=0)
            else:
                infer_params = [gate, up]

        else:
            # concat tensor
            infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(param))

        return infer_params

    def _post_process_params(self, params, convert_qkv_gate_up_by_simple_split=False):
        from megatron.core import mpu
        """
        For each param, if it is a tp-splited param, we all-gather from micro_dp group.
        """
        # here the params are in train tp format. we iterate params and all-gather
        # TODO(zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
        # In this way, all the params in the original memory_buffers and can be offload.
        micro_dp_size = get_micro_data_parallel_world_size()
        micro_dp_group = get_micro_data_parallel_group()

        for name, param in params:
            if tp_utils.is_tensor_parallel_param(param):
                # allocate a new tensor with proper size
                if micro_dp_size <= 1:
                    infer_params = [param]
                else:
                    infer_params = [torch.empty_like(param) for _ in range(micro_dp_size)]
                    torch.distributed.all_gather(infer_params, param, group=micro_dp_group)
                infer_params = self.default_tp_concat_fn(name, param, infer_params, self.model_config,
                                                         convert_qkv_gate_up_by_simple_split)
            else:
                infer_params = param
            converted_names, converted_params = convert_megatron_model_to_transformers_model(
                name,
                infer_params,
                self.model_config,
                mpu.get_tensor_model_parallel_world_size(),
                self.module.pp_models[0][0].config.num_query_groups,
                convert_qkv_gate_up_by_trunk_concat=False)
            for converted_name, infer_param in zip(converted_names, converted_params):
                yield converted_name, infer_param

    def __enter__(self):
        from megatron.core import mpu

        per_tensor_param = self.per_tensor_generator()
        self.inference_engine.resume_memory_occupation()

        self.inference_engine.update_weights_from_tensor(per_tensor_param, load_format=None)

        log_gpu_memory_usage('After load_weights sharding manager memory', logger=None)
        log_gpu_memory_usage('After delete params sharding manager memory', logger=None)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before SGLang offload in sharding manager', logger=logger)
        self.inference_engine.release_memory_occupation()
        log_gpu_memory_usage('After SGLang offload in sharding manager', logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        torch.cuda.empty_cache()


"""
Micro Data parallel group
"""


def get_micro_data_parallel_group():
    assert _MICRO_DATA_PARALLEL_GROUP is not None
    return _MICRO_DATA_PARALLEL_GROUP


def get_micro_data_parallel_world_size():
    return torch.distributed.get_world_size(group=get_micro_data_parallel_group())


def get_micro_data_parallel_rank():
    return torch.distributed.get_rank(group=get_micro_data_parallel_group())
