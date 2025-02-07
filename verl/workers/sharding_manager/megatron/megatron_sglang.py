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
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine SGLang.

Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

from ..base import BaseShardingManager
from .megatron_ppmodel import AllGatherPPModel
import torch
from torch import nn
import torch.distributed
from torch.distributed import new_group
from megatron.core import parallel_state as mpu
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
import verl.utils.megatron.tensor_parallel as tp_utils
from verl.utils.model import normalize_pp_vpp_params
# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
_MICRO_DATA_PARALLEL_GROUP = None

class MegatronSGLangShardingManager(BaseShardingManager):

    def __init__(self, module: AllGatherPPModel, inference_engine, model_config, layer_name_mapping):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.layer_name_mapping = layer_name_mapping

        # initialize micro_dp group for vllm inference
        global _MICRO_DATA_PARALLEL_GROUP
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        train_tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
        infer_tensor_parallel_size = vllm_ps.get_tensor_model_parallel_world_size()

        # TODO(sgm): this may not be true for FSDP -> vLLM
        assert infer_tensor_parallel_size <= train_tensor_parallel_size, \
            'Not implemented for infer_tp > train_tp'
        assert train_tensor_parallel_size % infer_tensor_parallel_size == 0

        micro_dp_size = train_tensor_parallel_size // infer_tensor_parallel_size
        num_micro_dp_groups = world_size // micro_dp_size
        assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
        for i in range(num_micro_dp_groups):
            ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
            group = new_group(ranks=ranks)
            if rank in ranks:
                _MICRO_DATA_PARALLEL_GROUP = group

    def default_tp_concat_fn(self, name, param, infer_params, model_config):
        """
        name: name of the parameter
        param: training parameters
        infer_params (List[torch.Tensor]): a list of parameters all-gathered from micro_dp_group
        model_config: huggingface model_config
        TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
        definition so that it is model-agnostic. If the model doesn't implement this function, 
        we can throw an error to force user disable TP HybridEngine.
        """

        if self.layer_name_mapping.get("qkv_layer_name") in name:
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
                q, k, v = infer_param.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)
            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)

            infer_params = torch.cat((q, k, v), dim=0)

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
            infer_params = torch.cat((gate, up), dim=0)

        else:
            # concat tensor
            infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(param))

        return infer_params

    def _post_process_params(self, params):
        """
        For each param, if it is a tp-splited param, we all-gather from micro_dp group.
        """
        # here the params are in train tp format. we iterate params and all-gather
        # TODO(zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
        # In this way, all the params in the original memory_buffers and can be offload.
        micro_dp_size = get_micro_data_parallel_world_size()
        micro_dp_group = get_micro_data_parallel_group()

        if micro_dp_size <= 1:
            return

        origin_params = {}
        for name in params.keys():
            param = params[name]
            if tp_utils.is_tensor_parallel_param(param):
                # allocate a new tensor with proper size
                infer_params = [torch.empty_like(param) for _ in range(micro_dp_size)]
                torch.distributed.all_gather(infer_params, param, group=micro_dp_group)
                infer_params = self.default_tp_concat_fn(name, param, infer_params, self.model_config)
                # replace with original param
                params[name] = infer_params
            origin_params[name] = param

        return origin_params

    def __enter__(self):
        # create a new cuda space for parameters not in this pp rank
        self.module.load_params_to_cuda()
        # broadcast the parameters from pp rank to other ranks
        self.module.allgather_params()
        # obtain name to parameters in pp/vpp
        params = self.module.get_all_params()

        # bind the params to inference engine
        self.params = normalize_pp_vpp_params(params=params,
                                              num_hidden_layers=self.model_config.num_hidden_layers,
                                              layer_name='layers')
        self.origin_params = self._post_process_params(self.params)
        self.inference_engine.sync_model_weights(self.params, load_format='megatron')

    def __exit__(self, exc_type, exc_value, traceback):
        # offload parameters doesn't belong to this pp rank
        self.module.offload_params_to_cpu()

        # FIXME(sgm): the best practice is to delete the cuda tensor
        # rebind the model weights, can be any cpu tensor
        if get_micro_data_parallel_world_size() > 1:
            for name in self.params.keys():
                self.params[name] = self.origin_params[name]

        # self.inference_engine.sync_model_weights(params)
        self.inference_engine.offload_model_weights()

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

    def preprocess_data(self, data: DataProto) -> DataProto:
        # prompts are identical for each training tp. We select for each inference tp
        micro_dp_size = get_micro_data_parallel_world_size()
        micro_dp_rank = get_micro_data_parallel_rank()

        # broadcast from tp=0 to other tp ranks
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())

        if micro_dp_size > 1:
            local_prompts = data.chunk(chunks=micro_dp_size)
            data = local_prompts[micro_dp_rank]

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        meta_info = data.meta_info
        # all gather batch among micro-dp groups
        micro_dp_size = get_micro_data_parallel_world_size()
        if micro_dp_size > 1:
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=get_micro_data_parallel_world_size(),
                                                group=get_micro_data_parallel_group(),
                                                dim=0)

        # all gather batch among pp group
        if meta_info.get('allgather_pp_output', True):
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=mpu.get_pipeline_model_parallel_world_size(),
                                                group=mpu.get_pipeline_model_parallel_group(),
                                                dim=0)
        return data
    
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