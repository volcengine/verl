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
from packaging.version import Version
import torch
import torch.distributed as dist

from torch import nn

from megatron.core import parallel_state as mpu
from megatron.core import DistributedDataParallel as LocalDDP
from megatron.core.transformer.module import Float16Module
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from verl.utils.megatron_utils import get_model, unwrap_model
from verl.utils.memory_buffer import (
    build_memory_buffer,
    build_memory_reference_from_module,
    get_weight_buffer_meta_from_module,
)


class AllGatherPPModel:

    def __init__(self, model_provider, use_distributed_optimizer=True) -> None:

        self._pp_group = mpu.get_pipeline_model_parallel_group()
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        self._model_chunk_size = self._vpp_size or 1

        # each one holds a list of model_chunks in this pp stage
        self._pp_models = [None] * self.pp_size

        rank_list = list(range(self.pp_size))
        # make current rank the last one to initialize
        rank_list[self.pp_rank], rank_list[-1] = rank_list[-1], rank_list[self.pp_rank]
        self._this_rank_models = None

        # store the parameter of each pp stage
        self.memory_buffers = [None] * self.pp_size
        for cur_pp_rank in rank_list:
            print(
                f'create pp model', f'torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, '
                f'reserved {torch.cuda.memory_reserved() / 1e9:.4f} GB')
            # since the last initialized rank is the current pp rank, after init, the pp rank is still correct
            mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
            if cur_pp_rank != self.pp_rank:
                models = get_model(model_provider, wrap_with_ddp=False, use_distributed_optimizer=False)
                models = nn.ModuleList(models)
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                self.pp_models[cur_pp_rank] = models
            else:
                # for regular model, we wrapped it with DDP
                models = get_model(model_provider,
                                   wrap_with_ddp=True,
                                   use_distributed_optimizer=use_distributed_optimizer)
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                self._this_rank_models = nn.ModuleList(models)
                self.pp_models[cur_pp_rank] = nn.ModuleList(unwrap_model(models, (torchDDP, LocalDDP)))

            self._build_param_buffer(cur_pp_rank)
            self._build_param_references(cur_pp_rank, maintain_weight=cur_pp_rank == self.pp_rank)

            # TODO: after binding to the memory buffer, we can load the checkpoint here
            if cur_pp_rank != self.pp_rank:
                for model in self.pp_models[cur_pp_rank]:
                    model.eval()
                self._offload_params_to_cpu(cur_pp_rank)

    def _build_param_buffer(self, pp_rank):
        """Build the parameter buffer in each pp rank"""
        if pp_rank == self._pp_rank:
            from verl.utils.memory_buffer import MemoryBuffer
            # The code here is very hard-coded, based on the following assumptions:
            # 1. `len(_this_rank_models) == 1`
            # 2. `_this_rank_models[0]` is a instance of `DistributedDataParallel` and `use_distributed_optimizer=True`
            # 3. Only bfloat16 data type is used in parameters
            source = self._this_rank_models[0].buffers[0].param_data
            self.memory_buffers[pp_rank] = {
                torch.bfloat16: MemoryBuffer(source.numel(), source.numel(), torch.bfloat16, source)
            }
        else:
            model = self.pp_models[pp_rank]
            weight_buffer_meta = get_weight_buffer_meta_from_module(model)
            self.memory_buffers[pp_rank] = build_memory_buffer(weight_buffer_meta)

    def _build_param_references(self, pp_rank, maintain_weight=False):
        if pp_rank == self._pp_rank:
            return
        model = self.pp_models[pp_rank]
        build_memory_reference_from_module(model, self.memory_buffers[pp_rank], maintain_weight=maintain_weight)

    def _load_params_to_cuda(self, pp_rank, to_empty=False):
        assert pp_rank != self.pp_rank, f"unexpected to load current pp rank [{pp_rank}] back to cuda"
        for buffer in self.memory_buffers[pp_rank].values():
            if not to_empty:
                buffer.data = buffer.data.to(torch.cuda.current_device(), non_blocking=True)
            else:
                buffer.data = torch.empty_like(buffer.data, device='cuda')
        # rebuild reference after loading to CUDA
        self._build_param_references(pp_rank)

    def _offload_params_to_cpu(self, pp_rank, to_empty=False):
        assert pp_rank != self.pp_rank, f"unexpected to offload current pp rank [{pp_rank}] to cpu"
        for buffer in self.memory_buffers[pp_rank].values():
            if not to_empty:
                # offload the whole memory buffer to CPU
                buffer.data = buffer.data.to('cpu', non_blocking=True)
            else:
                buffer.data = torch.empty_like(buffer.data, device='cpu')
        self._build_param_references(pp_rank)

    def load_params_to_cuda(self, to_empty=False):
        """load all model params to cuda"""
        for cur_pp_rank in range(self.pp_size):
            if cur_pp_rank != self.pp_rank:
                self._load_params_to_cuda(cur_pp_rank, to_empty=to_empty)

    def allgather_params(self):
        """allgather params of all pp ranks. Return a list of handles"""
        for cur_pp_rank in range(self.pp_size):
            global_src = dist.get_global_rank(group=self.pp_group, group_rank=cur_pp_rank)

            # NOTE(sgm): the async op may cause memory leakage of the memory_buffer/pp_models

            for _, param in sorted(self.pp_models[cur_pp_rank].named_parameters()):
                dist.broadcast(tensor=param.data, src=global_src, group=self.pp_group, async_op=False)

    def forward(self, *inputs, **kwargs):
        try:
            prev_output = None
            for cur_chunk_rank in range(self._model_chunk_size):
                if self._vpp_size:
                    mpu.set_virtual_pipeline_model_parallel_rank(cur_chunk_rank)

                for cur_pp_rank in range(self.pp_size):
                    mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(prev_output)
                    ret = self.pp_models[cur_pp_rank][cur_chunk_rank](*inputs, **kwargs)
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(None)
                    prev_output = ret
        finally:
            if self._vpp_size:
                mpu.set_virtual_pipeline_model_parallel_rank(0)
            mpu.set_pipeline_model_parallel_rank(self.pp_rank)
        return ret

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def eval(self):
        for model in self.pp_models[self.pp_rank]:
            model.eval()

    def train(self):
        for model in self.pp_models[self.pp_rank]:
            model.train()

    def offload_params_to_cpu(self, to_empty=False):
        """offload params of models that are not of current pp rank to cpu"""
        for cur_pp_rank in range(self.pp_size):
            if cur_pp_rank != self.pp_rank:
                self._offload_params_to_cpu(cur_pp_rank, to_empty=to_empty)

    def get_all_params(self):
        """Get all the parameters of the models in all pp ranks

        Returns:
            params: List[List[Dict[str, Tensor]]]: a list of parameters in all pp, where each is a list of dict
                tensors of each model chunk

        """
        params = []
        for pp_rank in range(self.pp_size):
            params.append([])
            for model_chunk_idx in range(len(self.pp_models[pp_rank])):
                params[pp_rank].append({})
                pp_model = self.pp_models[pp_rank][model_chunk_idx]
                pp_model = unwrap_model(pp_model, ((torchDDP, LocalDDP, Float16Module)))  # not use Float16Module
                for name, param in pp_model.named_parameters():
                    # NOTE(gh) workaround: should not get lora params for inference
                    if 'lora' in name:
                        continue
                    params[pp_rank][model_chunk_idx][name] = param

        return params

    def update_this_rank_models(self, new_models):
        self._this_rank_models = new_models
        self._pp_models[self.pp_rank] = unwrap_model(new_models, (torchDDP, LocalDDP))

    @property
    def this_rank_models(self):
        return self._this_rank_models

    @property
    def pp_size(self):
        return self._pp_size

    @property
    def pp_rank(self):
        return self._pp_rank

    @property
    def pp_group(self):
        return self._pp_group

    @property
    def pp_models(self):
        return self._pp_models


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

from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
import verl.utils.megatron.tensor_parallel as tp_utils
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.third_party.vllm import LLM
from verl.utils.model import normalize_pp_vpp_params
# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
_MICRO_DATA_PARALLEL_GROUP = None


class MegatronVLLMShardingManager(BaseShardingManager):

    def __init__(self, module: AllGatherPPModel, inference_engine: LLM, model_config, layer_name_mapping):
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

        origin_params = {}
        for name in params.keys():
            param = params[name]
            if tp_utils.is_tensor_parallel_param(param):
                # allocate a new tensor with proper size
                if micro_dp_size <= 1:
                    infer_params = [param]
                else:
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
        if self.origin_params is not None:
            for name, param in self.origin_params.items():
                self.params[name] = param

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
