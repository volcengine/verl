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

    def __init__(self, model_provider) -> None:

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
                models = get_model(model_provider, wrap_with_ddp=False)
                models = nn.ModuleList(models)
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                self.pp_models[cur_pp_rank] = models
            else:
                # for regular model, we wrapped it with DDP
                models = get_model(model_provider)
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
        model = self.pp_models[pp_rank]
        weight_buffer_meta = get_weight_buffer_meta_from_module(model)
        self.memory_buffers[pp_rank] = build_memory_buffer(weight_buffer_meta)

    def _build_param_references(self, pp_rank, maintain_weight=False):
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
            for memory_buffer in self.memory_buffers[cur_pp_rank].values():
                dist.broadcast(tensor=memory_buffer.data, src=global_src, group=self.pp_group, async_op=False)

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
