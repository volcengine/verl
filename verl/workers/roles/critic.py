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
The main entry point to run the PPO algorithm
"""

import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Union

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.debug.performance import _timer, reduce_timing
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.engine.fsdp import FSDPEngine
from verl.utils.seqlen_balancing import (get_reverse_idx,
                                         rearrange_micro_batches)
import itertools

from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import (gather_outpus_and_unpad,
                                ulysses_pad_and_slice_inputs)
from verl.workers.critic import BasePPOCritic

if is_cuda_available:
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         rearrange, unpad_input)
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import (
        index_first_axis, pad_input, rearrange, unpad_input)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


# def create_device_mesh(world_size, fsdp_size):
#     if fsdp_size < 0 or fsdp_size >= world_size:
#         device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
#     else:
#         device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
#     return device_mesh


# def get_sharding_strategy(device_mesh):
#     from torch.distributed.fsdp import ShardingStrategy

#     if device_mesh.ndim == 1:
#         sharding_strategy = ShardingStrategy.FULL_SHARD
#     elif device_mesh.ndim == 2:
#         sharding_strategy = ShardingStrategy.HYBRID_SHARD
#     else:
#         raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
#     return sharding_strategy



class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if is_cuda_available else "hccl")
        self.config = config

        # # build device mesh for Ulysses Sequence Parallel
        # world_size = torch.distributed.get_world_size()
        # from torch.distributed.device_mesh import init_device_mesh

        # fsdp_size = self.config.model.fsdp_config.fsdp_size
        # self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        # self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        # dp = world_size // self.ulysses_sequence_parallel_size
        # if self.ulysses_sequence_parallel_size > 1:
        #     self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        # self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # # set FSDP offload params
        # self._is_offload_param = self.config.model.fsdp_config.param_offload
        # self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.forward_micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
        # self._is_lora = self.config.model.get("lora_rank", 0) > 0

        self.engine = FSDPEngine(self.config)
        self.device_name = get_device_name()

        # def preprocess_fn_with_rmpad(batch, ctx):
        #     ctx["response_length"] = batch["responses"].size(-1)

        #     inputs = {}
        #     if "multi_modal_inputs" in batch.keys():
        #         for key in batch["multi_modal_inputs"][0].keys():
        #             inputs[key] = torch.cat([inputs[key] for inputs in batch["multi_modal_inputs"]], dim=0)

        #     input_ids = batch["input_ids"]
        #     attention_mask = batch["attention_mask"]
        #     position_ids = batch["position_ids"]
        #     if position_ids.dim() == 3:  # qwen2vl mrope
        #         position_ids = position_ids.transpose(0, 1)

        #     input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        #     input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        #     # unpad the position_ids to align the rotary
        #     if position_ids.dim() == 3:
        #         position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
        #     else:
        #         position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

        #     # pad and slice the inputs if sp > 1
        #     if self.ulysses_sequence_parallel_size > 1:
        #         input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

        #     inputs["input_ids"] = input_ids_rmpad
        #     inputs["attention_mask"] = attention_mask
        #     inputs["position_ids"] = position_ids_rmpad

        #     ctx["pad_size"] = pad_size
        #     ctx["indices"] = indices
        #     ctx["seqlen"] = seqlen
        #     ctx["batch"] = batch

        #     return inputs, ctx

        # def postprocess_fn_with_rmpad(outputs, ctx):
        #     response_length = ctx["response_length"]
        #     if hasattr(self.critic_module, "v_head"):
        #         # For trl.AutoModelForCausalLMWithValueHead
        #         values_rmpad = output[2].squeeze(0).unsqueeze(-1)
        #     else:
        #         values_rmpad = output.logits
        #         values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

        #     # gather output if sp > 1
        #     if self.ulysses_sequence_parallel_size > 1:
        #         values_rmpad = gather_outpus_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

        #     # pad it back
        #     values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
        #     values = values[:, -response_length - 1 : -1]
        #     return values


        # def postprocess_fn(outputs, ctx):
        #     return preds, ctx


        # def loss_fn(data, preds, ctx):
        #     return loss, out_ctx



    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.init_model_and_optimizer()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        def preprocess_fn_without_rmpad(batch, ctx):
            ctx["response_length"] = batch["responses"].size(-1)

            inputs = {}
            if "multi_modal_inputs" in batch.keys():
                for key in batch["multi_modal_inputs"][0].keys():
                    inputs[key] = torch.cat([inputs[key] for inputs in batch["multi_modal_inputs"]], dim=0)

            inputs["input_ids"] = batch["input_ids"]
            inputs["attention_mask"] = batch["attention_mask"]
            position_ids = batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            inputs["position_ids"] = position_ids
            return inputs, ctx


        def postprocess_fn_without_rmpad(outputs, ctx):
            response_length = ctx["response_length"]
            use_value_head_model = ctx["use_value_head_model"]
            if use_value_head_model:
                # For trl.AutoModelForCausalLMWithValueHead
                values = outputs[2]
            else:
                values = outputs.logits
            values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values, ctx

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        assert self.use_remove_padding == False
        self.engine.set_preprocess_fn(preprocess_fn_without_rmpad)
        self.engine.set_postprocess_fn(postprocess_fn_without_rmpad)
        

        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            
            micro_batch_size = data.meta_info["micro_batch_size"]
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

            if has_multi_modal_inputs:
                num_micro_batches = data.batch.batch_size[0] // micro_batch_size
                non_tensor_select_keys = ["multi_modal_inputs"]
                micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
            elif use_dynamic_bsz:
                # split using dynamic bsz
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            else:
                micro_batches = batch.split(micro_batch_size)

            values_lst = []
            for micro_batch in micro_batches:
                if isinstance(micro_batch, DataProto):
                    micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    # TODO: should not access module in the engine
                    use_value_head_model = hasattr(self.engine.critic_module, "v_head")
                    ctx = {"use_value_head_model": use_value_head_model}
                    values = self.engine.forward_backward_step(micro_batch, ctx, forward_only=True)
                values_lst.append(values)
            values = torch.concat(values_lst, dim=0)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                values = values[revert_indices]
                
            responses = data.batch["responses"]
            attention_mask = data.batch["attention_mask"]
            response_length = responses.size(1)
            response_mask = attention_mask[:, -response_length:]
            values = values * response_mask # Only action tokens have values
            output = DataProto.from_dict(tensors={"values": values})
            output = self.engine.unshard_data(output)

        output = output.to("cpu")
        raise ValueError
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_torch_device().current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_torch_device().current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)

