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
import torch.distributed as dist
from megatron.core import DistributedDataParallel as LocalDDP
from megatron.core import parallel_state as mpu
from megatron.core.transformer.module import Float16Module
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

import verl.utils.megatron.tensor_parallel as tp_utils
from verl import DataProto
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.megatron_utils import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    convert_megatron_model_to_transformers_model,
    get_model,
    unwrap_model,
)
from verl.utils.memory_buffer import (
    build_memory_buffer,
    build_memory_reference_from_module,
    get_weight_buffer_meta_from_module,
)
from verl.utils.model import normalize_model_name
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AllGatherPPModel:
    def __init__(self, model_provider, use_distributed_optimizer=True) -> None:
        print(
            "[WARNING] This class is deprecated and will no longer be supported. \
Consider using the `MegatronPPOActor` class directly as a replacement."
        )
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
                "create pp model",
                f"torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, reserved {torch.cuda.memory_reserved() / 1e9:.4f} GB",
            )
            # since the last initialized rank is the current pp rank, after init, the pp rank is still correct
            mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
            if cur_pp_rank != self.pp_rank:
                models = get_model(model_provider, wrap_with_ddp=False, use_distributed_optimizer=False)
                models = nn.ModuleList(models)
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                self.pp_models[cur_pp_rank] = models
            else:
                # for regular model, we wrapped it with DDP
                models = get_model(model_provider, wrap_with_ddp=True, use_distributed_optimizer=use_distributed_optimizer)
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
            self.memory_buffers[pp_rank] = {torch.bfloat16: MemoryBuffer(source.numel(), source.numel(), torch.bfloat16, source)}
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
                buffer.data = torch.empty_like(buffer.data, device="cuda")
        # rebuild reference after loading to CUDA
        self._build_param_references(pp_rank)

    def _offload_params_to_cpu(self, pp_rank, to_empty=False):
        assert pp_rank != self.pp_rank, f"unexpected to offload current pp rank [{pp_rank}] to cpu"
        for buffer in self.memory_buffers[pp_rank].values():
            if not to_empty:
                # offload the whole memory buffer to CPU
                buffer.data = buffer.data.to("cpu", non_blocking=True)
            else:
                buffer.data = torch.empty_like(buffer.data, device="cpu")
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
                    if "lora" in name:
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
- Before inference, broadcast the parameters of the current pp rank 
   to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
# NOTICE: in new version of vLLM, We need to all-gather all tp rank's model weights
# For code reuse, we directly assign Megatron's TENSOR_MODEL_PARALLEL_GROUP to this
_MICRO_DATA_PARALLEL_GROUP = None


class MegatronVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: LLM,
        model_config,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        module: AllGatherPPModel = None,
    ):
        from megatron.core import parallel_state as mpu

        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.module = module
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

    def per_tensor_generator(self, convert_qkv_gate_up_by_simple_split=True):
        """
        convert_qkv_gate_up_by_simple_split is a parameter affected by the vLLM version.
        """
        from megatron.core import parallel_state as mpu

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        vpp_size = len(self.actor_module)

        all_gather_group = self.train_tp_group
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
        torch.distributed.all_gather_object(object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group())
        layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

        gen_func = tensor_generator()

        # lazy load tensor for full model
        for cur_pp_rank, scan_vpp_idx, idx, name in layer_list_meta:
            if self.model_config.tie_word_embeddings and ("output_layers" in name):
                import warnings

                warnings.warn("Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2)
                continue
            if cur_pp_rank == pp_rank:
                try:
                    cur_name, cur_tensor = next(gen_func)
                except StopIteration:
                    cur_name, cur_tensor = None, None
                cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, pp_size, vpp_size, self.model_config.num_hidden_layers)
            else:
                cur_tensor, cur_name = None, None

            # pp broadcast model tensor and name
            cur_name = broadcast_str_from_megatron_pp(cur_name)
            broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

            # (xya): this is a hack to fix the name of the parameters
            while cur_name.startswith("module."):
                cur_name = cur_name[len("module.") :]

            # EP
            if ".mlp.experts.linear_fc" in cur_name and self.train_ep_size > 1:
                num_experts = self.weight_converter.mcore_config.num_moe_experts
                num_experts_per_rank = num_experts // self.train_ep_size
                infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(self.train_ep_size)]
                torch.distributed.all_gather(infer_params, broad_pp_tensor, group=self.train_ep_group)

                name_prefix, local_expert_id = cur_name.split(".weight")
                local_expert_id = int(local_expert_id)
                global_expert_ids = [num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(self.train_ep_size)]
                global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

                for name, param in zip(global_expert_names, infer_params):
                    if self.train_etp_size > 1:
                        # gather etp
                        etp_params = [torch.empty_like(param) for _ in range(self.train_etp_size)]
                        torch.distributed.all_gather(etp_params, param, group=self.train_etp_group)
                        params = etp_params
                    else:
                        params = [param]

                    merge_params = self.default_tp_concat_fn(name, broad_pp_tensor, params, self.model_config, convert_qkv_gate_up_by_simple_split)
                    if not isinstance(merge_params, list):
                        merge_params = [merge_params]
                    converted_names, converted_params = self.weight_converter.convert_param(name, merge_params)

                    yield from zip(converted_names, converted_params)
                continue

            # tp all gather
            if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
                # allocate a new tensor with proper size
                if all_gather_group_size <= 1:
                    infer_params = [broad_pp_tensor]
                else:
                    infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                    torch.distributed.all_gather(infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group())
                infer_params = self.default_tp_concat_fn(cur_name, broad_pp_tensor, infer_params, self.model_config, convert_qkv_gate_up_by_simple_split)
            else:
                infer_params = broad_pp_tensor

            if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
                converted_names, converted_params = convert_megatron_model_to_transformers_model(
                    cur_name,
                    infer_params,
                    self.model_config,
                    self.train_tp_size,
                    0,  # no impact
                    convert_qkv_gate_up_by_trunk_concat=False,
                )  # defualt false
            else:
                if not isinstance(infer_params, list):
                    infer_params = [infer_params]
                converted_names, converted_params = self.weight_converter.convert_param(cur_name, infer_params)

            yield from zip(converted_names, converted_params)

    def default_tp_concat_fn(self, name, param, infer_params, model_config, convert_qkv_gate_up_by_simple_split=False):
        """
        name: name of the parameter
        param: training parameters
        infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered
          from train tp group (vllm 0.8.2) or micro-dp group (vllm <= 0.6.3)
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
            assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0, f"param '{name}' shape '{infer_params[0].shape}' dim0 is not divisible by {num_q_per_kv + 2}"
            kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
            for infer_param in infer_params:
                num_query_groups_per_partition = model_config.num_key_value_heads // self.train_tp_size
                for chunk in infer_param.chunk(num_query_groups_per_partition):
                    split_size = [
                        kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                    ]
                    q, k, v = chunk.split(split_size)
                    q_lst.append(q)
                    k_lst.append(k)
                    v_lst.append(v)
            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)
            infer_params = torch.cat((q, k, v), dim=0) if not convert_qkv_gate_up_by_simple_split else [q, k, v]

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
            infer_params = torch.cat((gate, up), dim=0) if not convert_qkv_gate_up_by_simple_split else [gate, up]

        elif "mlp.experts.linear_fc2.weight" in name:  # moe
            infer_params = torch.cat(infer_params, dim=1)

        else:
            # concat tensor
            infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(param))

        return infer_params

    def _post_process_params(self, params, convert_qkv_gate_up_by_simple_split=False):
        """
        For each param, if it is a tp-splited param, we all-gather from train tp group
        """
        # here the params are in train tp format. we iterate params and all-gather
        # TODO(zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
        # In this way, all the params in the original memory_buffers and can be offload.
        all_gather_group = self.train_tp_group
        all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)

        for name, param in params:
            if tp_utils.is_tensor_parallel_param(param):
                # allocate a new tensor with proper size
                if all_gather_group_size <= 1:
                    infer_params = [param]
                else:
                    infer_params = [torch.empty_like(param) for _ in range(all_gather_group_size)]
                    torch.distributed.all_gather(infer_params, param, group=all_gather_group)
                infer_params = self.default_tp_concat_fn(name, param, infer_params, self.model_config, convert_qkv_gate_up_by_simple_split)
            else:
                infer_params = param
            if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
                converted_names, converted_params = convert_megatron_model_to_transformers_model(
                    name,
                    infer_params,
                    self.model_config,
                    self.train_tp_size,
                    self.module.pp_models[0][0].config.num_query_groups,
                    convert_qkv_gate_up_by_trunk_concat=False,
                )
            else:
                if not isinstance(infer_params, list):
                    infer_params = [infer_params]
                converted_names, converted_params = self.weight_converter.convert_param(name, infer_params)
            yield from zip(converted_names, converted_params)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            per_tensor_param = self.per_tensor_generator(convert_qkv_gate_up_by_simple_split=False)
            self.inference_engine.sync_model_weights(per_tensor_param, load_format="megatron")
        else:
            # > 0.7.2
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()
            per_tensor_param = self.per_tensor_generator()
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            loaded_params = model.load_weights(per_tensor_param)
            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            logger.info(info)

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

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
