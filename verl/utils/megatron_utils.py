# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Pretrain utilities."""
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import ModelParallelConfig
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model
from omegaconf import DictConfig

from verl.utils.memory_buffer import build_memory_reference_from_module
from verl.utils.torch_dtypes import PrecisionType


def get_model_config(model):
    return get_attr_wrapped_model(model, 'config', allow_none=False)


def get_model(model_provider_func,
              model_type=ModelType.encoder_or_decoder,
              wrap_with_ddp=True,
              use_distributed_optimizer=True):
    """Build the model."""
    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(pre_process=pre_process,
                                        post_process=post_process,
                                        add_encoder=add_encoder,
                                        add_decoder=add_decoder)
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
                  mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(),
                  sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])),
              flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    config: TransformerConfig = get_model_config(model[0])
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=DistributedDataParallelConfig(
                    overlap_grad_reduce=False,
                    use_distributed_optimizer=use_distributed_optimizer,
                    grad_reduce_in_fp32=True,  # [old] accumulate_allreduce_grads_in_fp32=True,
                ))
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


from transformers import PretrainedConfig


def convert_config(hf_config: PretrainedConfig, megatron_config) -> TransformerConfig:
    print(f'megatron config {megatron_config}')
    dt = PrecisionType.to_dtype(megatron_config.params_dtype)
    print(f'pipeline_dtype=megatron_config {dt}')
    if "Qwen2ForCausalLM" in hf_config.architectures:
        qkv_bias = True
    else:
        qkv_bias = getattr(hf_config, 'attention_bias', False)
    overlap_p2p_comm = mpu.get_virtual_pipeline_model_parallel_world_size(
    ) is not None and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    batch_p2p_comm = False
    transformer_config = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        #    max_position_embeddings=hf_config.max_position_embeddings,
        activation_func=F.silu,
        normalization='RMSNorm',
        #    rotary_percent=False, # default,
        gated_linear_unit=True,  # for llama
        use_cpu_initialization=True,
        apply_residual_connection_post_layernorm=False,  # check what's this mean
        add_bias_linear=False,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        pipeline_dtype=dt,
        params_dtype=dt,
        sequence_parallel=mpu.get_tensor_model_parallel_world_size() > 1,
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        moe_token_dispatcher_type="alltoall",
        attention_dropout=hf_config.attention_dropout,
        hidden_dropout=getattr(hf_config, 'hidden_dropout', 0.0),
        add_qkv_bias=qkv_bias,
        attention_backend=AttnBackend.flash,
        bf16=dt is torch.bfloat16)

    return transformer_config


def init_megatron_optim_config(optim_config: Dict) -> OptimizerConfig:
    config = OptimizerConfig(
        optimizer='adam',
        lr=optim_config.get('lr'),
        clip_grad=optim_config.get('clip_grad'),
        weight_decay=1e-2,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    return config


def mcore_model_parallel_config(
    sequence_parallel: bool,
    params_dtype: torch.dtype,
) -> ModelParallelConfig:
    return ModelParallelConfig(
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        bf16=True,
        fp16=False,
        timers=None)


def offload_megatron_param_and_grad(module_list: nn.ModuleList, offload_grad=False, hybrid_engine=None):
    if hybrid_engine is not None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        for buffer in hybrid_engine.memory_buffers[pp_rank].values():
            buffer.data = buffer.data.to('cpu', non_blocking=True)
        build_memory_reference_from_module(module_list, hybrid_engine.memory_buffers[pp_rank], maintain_weight=True)
    else:
        for module in module_list:
            for _, param in module.named_parameters():
                param.data = param.data.to('cpu', non_blocking=True)
                if offload_grad and param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_megatron_param_and_grad(module_list: nn.ModuleList, device_id, load_grad=False, hybrid_engine=None):
    if hybrid_engine is not None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        for buffer in hybrid_engine.memory_buffers[pp_rank].values():
            buffer.data = buffer.data.to(device_id, non_blocking=True)
        build_memory_reference_from_module(module_list, hybrid_engine.memory_buffers[pp_rank], maintain_weight=True)
    else:
        for module in module_list:
            for _, param in module.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if load_grad and param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_model_checkpoint_path(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    return os.path.join(checkpoint_path, "model")


def get_hf_model_checkpoint_path(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    return os.path.join(checkpoint_path, "huggingface")


def get_optimizer_checkpoint_path(checkpoint_path, use_distributed_optimizer=True):
    os.makedirs(os.path.join(checkpoint_path, "optim"), exist_ok=True)
    if not use_distributed_optimizer:
        return os.path.join(checkpoint_path, "optim", "optim.pt")
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    #TODO: support ep
    return os.path.join(checkpoint_path, f"optim", f"distrib_optim_pp{pp_rank}_tp{tp_rank}_cp{cp_rank}_dp{dp_rank}.pt")


def get_rng_states_checkpoint_path(checkpoint_path, only_rank0_save=True):
    # save rng states cause interrupts
    os.makedirs(os.path.join(checkpoint_path, "rng_states"), exist_ok=True)
    if only_rank0_save:
        return os.path.join(checkpoint_path, f'rng_states', "rng_states.pt")
    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    return os.path.join(checkpoint_path, f'rng_states',
                        f"rng_states_pp{pp_rank}_tp{tp_rank}_cp{cp_rank}_dp{dp_rank}.pt")


def convert_megatron_model_to_transformers_model(name,
                                                 param,
                                                 config: PretrainedConfig,
                                                 tp_size: int,
                                                 num_query_groups: int,
                                                 convert_qkv_gate_up_by_trunk_concat=False):
    """Convert megatron model to transformers model."""
    new_params = {}

    def convert_qkv_shard(full_tensor, q_name, k_name, v_name):
        nonlocal config
        nonlocal tp_size
        nonlocal num_query_groups

        q_shard_list = []
        k_shard_list = []
        v_shard_list = []
        hidden_size_per_head = config.hidden_size // config.num_attention_heads

        if config.num_key_value_heads >= tp_size:
            q_size_tp = config.hidden_size // tp_size
            kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
            total_size = q_size_tp + 2 * kv_size_tp
            for i in range(tp_size):
                num_query_groups_per_partition = num_query_groups // tp_size
                qkv_part = full_tensor[i * total_size:(i + 1) * total_size]
                q_size_chunk = q_size_tp // num_query_groups_per_partition
                kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                    q_part = qkv_part_chunk[:q_size_chunk]
                    k_part = qkv_part_chunk[q_size_chunk:q_size_chunk + kv_size_chunk]
                    v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk:]
                    q_shard_list.append(q_part)
                    k_shard_list.append(k_part)
                    v_shard_list.append(v_part)
        else:
            q_size_tp = config.hidden_size // tp_size
            kv_size_tp = hidden_size_per_head
            total_size = q_size_tp + 2 * kv_size_tp
            for i in range(tp_size):
                num_query_groups_per_partition = num_query_groups // tp_size
                qkv_part = full_tensor[i * total_size:(i + 1) * total_size]
                q_size_chunk = q_size_tp // num_query_groups_per_partition
                kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                    q_part = qkv_part_chunk[:q_size_chunk]
                    k_part = qkv_part_chunk[q_size_chunk:q_size_chunk + kv_size_chunk]
                    v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk:]
                    q_shard_list.append(q_part)
                    if i * config.num_key_value_heads % tp_size == 0:
                        k_shard_list.append(k_part)
                        v_shard_list.append(v_part)

        new_params[q_name] = torch.cat(q_shard_list, dim=0)
        new_params[k_name] = torch.cat(k_shard_list, dim=0)
        new_params[v_name] = torch.cat(v_shard_list, dim=0)

    def convert_gate_up_shard(full_tensor, gate_name, up_name):
        nonlocal config
        nonlocal tp_size

        intermediate_size_tp = config.intermediate_size // tp_size
        gate_weight_list = []
        up_weight_list = []
        for i in range(tp_size):
            gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)]
            gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
            up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
            gate_weight_list.append(gate_weight_tp)
            up_weight_list.append(up_weight_tp)

        new_params[gate_name] = torch.cat(gate_weight_list, dim=0)
        new_params[up_name] = torch.cat(up_weight_list, dim=0)

    if name == 'embedding.word_embeddings.weight':
        new_params['model.embed_tokens.weight'] = param
    elif 'self_attention' in name:
        splitted_name = name.split('.')
        layer_number = splitted_name[2]
        component = splitted_name[4]
        param_type = splitted_name[5]
        if component == 'linear_proj':
            new_params[f'model.layers.{layer_number}.self_attn.o_proj.weight'] = param
        elif component == 'linear_qkv' and not isinstance(param, list):
            if param_type == 'layer_norm_weight':
                new_params[f'model.layers.{layer_number}.input_layernorm.weight'] = param
            else:
                if convert_qkv_gate_up_by_trunk_concat:
                    convert_qkv_shard(param, f'model.layers.{layer_number}.self_attn.q_proj.{param_type}',
                                      f'model.layers.{layer_number}.self_attn.k_proj.{param_type}',
                                      f'model.layers.{layer_number}.self_attn.v_proj.{param_type}')
                else:
                    new_params[f'model.layers.{layer_number}.self_attn.qkv_proj.{param_type}'] = param
        else:
            assert isinstance(param, list) and len(param) == 3
            assert param_type == 'weight' or param_type == 'bias'
            new_params[f'model.layers.{layer_number}.self_attn.q_proj.{param_type}'] = param[0]
            new_params[f'model.layers.{layer_number}.self_attn.k_proj.{param_type}'] = param[1]
            new_params[f'model.layers.{layer_number}.self_attn.v_proj.{param_type}'] = param[2]
    elif 'mlp' in name:
        splitted_name = name.split('.')
        layer_number = splitted_name[2]
        component = splitted_name[4]
        param_type = splitted_name[5]
        if component == 'linear_fc1' and not isinstance(param, list):
            if param_type == 'layer_norm_weight':
                new_params[f'model.layers.{layer_number}.post_attention_layernorm.weight'] = param
            elif param_type == 'weight':
                if convert_qkv_gate_up_by_trunk_concat:
                    convert_gate_up_shard(param, f'model.layers.{layer_number}.mlp.gate_proj.weight',
                                          f'model.layers.{layer_number}.mlp.up_proj.weight')
                else:
                    new_params[f'model.layers.{layer_number}.mlp.gate_up_proj.weight'] = param
        elif component == 'linear_fc1' and isinstance(param, list):
            assert len(param) == 2
            assert param_type == 'weight' or param_type == 'bias'
            new_params[f'model.layers.{layer_number}.mlp.gate_proj.weight'] = param[0]
            new_params[f'model.layers.{layer_number}.mlp.up_proj.weight'] = param[1]
        elif component == 'linear_fc2':
            new_params[f'model.layers.{layer_number}.mlp.down_proj.weight'] = param
    elif name == "decoder.final_layernorm.weight":
        new_params['model.norm.weight'] = param
    elif name == "output_layer.weight":
        new_params["lm_head.weight"] = param
    else:
        raise ValueError(f"Unknown param name: {name}")
    return new_params.keys(), new_params.values()