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
from typing import Any, Dict
import time
from omegaconf import DictConfig
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.memory_buffer import build_memory_reference_from_module
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
# from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
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
    config = get_model_config(model[0])
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        model = [
            DDP(config=config,
                module=model_chunk,
                data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                accumulate_allreduce_grads_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=True,
                disable_bucketing=(model_chunk_idx > 0)) for (model_chunk_idx, model_chunk) in enumerate(model)
        ]
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # if args.data_parallel_random_init:
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
    dt = PrecisionType.to_dtype(megatron_config['param_dtype'])
    print(f'pipeline_dtype=megatron_config {dt}')
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
        pipeline_dtype=PrecisionType.to_dtype(megatron_config['param_dtype']),
        params_dtype=PrecisionType.to_dtype(megatron_config['param_dtype']),
        sequence_parallel=megatron_config['sequence_parallel_enabled'],
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        bf16=PrecisionType.to_dtype(megatron_config['param_dtype']) is torch.bfloat16)
    if torch.distributed.get_rank() == 0:
        print(f'tensor_parallel_size={transformer_config.tensor_model_parallel_size} \n \
                pipeline_model_parallel_size={transformer_config.pipeline_model_parallel_size} \n \
                virtual_pipeline_model_parallel_size={transformer_config.virtual_pipeline_model_parallel_size} \n \
                pipeline_dtype={transformer_config.pipeline_dtype} \n \
                params_dtype={transformer_config.params_dtype} \n \
                sequence_parallel={transformer_config.sequence_parallel} \n \
                variable_seq_lengths={transformer_config.variable_seq_lengths} \n \
                masked_softmax_fusion={transformer_config.masked_softmax_fusion} \n ')

    return transformer_config


# from megatron.core.optimizer import OptimizerConfig

from verl.utils.megatron.optimizer_config import OptimizerConfig


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


from megatron.core import ModelParallelConfig


def init_model_parallel_config(config: DictConfig) -> ModelParallelConfig:
    # TODO(sgm): check how to disable megatron timers
    timers = FakeTimers()
    return ModelParallelConfig(tensor_model_parallel_size=config.get('tensor_model_parallel_size'),
                               pipeline_model_parallel_size=config.get('pipeline_model_parallel_size'),
                               virtual_pipeline_model_parallel_size=config.get('virtual_pipeline_model_parallel_size'),
                               sequence_parallel=config.get('sequence_parallel'),
                               params_dtype=PrecisionType.to_dtype(config.get('param_dtype')),
                               pipeline_dtype=PrecisionType.to_dtype(config.get('param_dtype')),
                               bf16=True,
                               fp16=False,
                               timers=timers)


class FakeTimers:
    """Disable All Megatron Timing with FakeTimers"""

    def __init__(self):
        from megatron.timers import DummyTimer
        self.dummy_timer = DummyTimer()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.dummy_timer


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
