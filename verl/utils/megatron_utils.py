# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import gc
import inspect
import os
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config
from transformers import PretrainedConfig

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import local_mkdir_safe


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
    override_ddp_config=None,
):
    """Build the model."""
    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert model_type != ModelType.encoder_and_decoder, (
            "Interleaved schedule not supported for model with both encoder and decoder"
        )
        model = []
        has_vp_stage = inspect.signature(mpu.is_pipeline_first_stage).parameters.get("vp_stage", None) is not None
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            extra_kwargs = {} if not has_vp_stage else {"ignore_virtual": False, "vp_stage": i}
            pre_process = mpu.is_pipeline_first_stage(**extra_kwargs)
            post_process = mpu.is_pipeline_last_stage(**extra_kwargs)
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process, vp_stage=i)
            this_model.model_type = model_type
            model.append(this_model)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        assert model_type != ModelType.encoder_and_decoder, "Model type encoder_and_decoder is not supported"
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, (
                    "Split rank needs to be specified for model with both encoder and decoder"
                )
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder
            )
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
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        for model_module in model:
            model_module.to(f"{get_device_name()}:{get_device_id()}")

    # Fp16 conversion.
    config: TransformerConfig = get_model_config(model[0])
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        ddp_config_dict = {
            "use_distributed_optimizer": use_distributed_optimizer,
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
        }
        if override_ddp_config is not None:
            ddp_config_dict.update(override_ddp_config)
        ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=ddp_config,
            )
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model


@dataclass
class McoreModuleWrapperConfig:
    """Configuration for Mcore module wrapper."""

    is_value_model: bool = False
    share_embeddings_and_output_weights: bool = False
    wrap_with_ddp: bool = True
    use_distributed_optimizer: bool = True


def make_megatron_module(
    wrap_config: McoreModuleWrapperConfig,
    tf_config: TransformerConfig,
    hf_config: PretrainedConfig,
    bridge: Any = None,
    provider: Any = None,
    override_model_config: dict[str, Any] = None,
    override_ddp_config: dict[str, Any] = None,
    peft_cls: Any = None,
    peft_config: Any = None,
):
    if override_model_config is None:
        override_model_config = {}

    if provider is None:
        from verl.models.mcore.mbridge import freeze_moe_router, make_value_model

        value_model_hook = make_value_model
    else:
        from verl.models.mcore.bridge import freeze_moe_router, make_value_model

        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        value_model_hook = make_value_model(hidden_size, provider.sequence_parallel)

    post_model_creation_callbacks = []
    if wrap_config.is_value_model:
        post_model_creation_callbacks.append(value_model_hook)
    if override_model_config.get("moe_config", {}).get("freeze_moe_router", False):
        post_model_creation_callbacks.append(freeze_moe_router)
    if provider is not None:
        # When using PEFT with Megatron-Bridge, we must apply PEFT transformation
        # BEFORE wrapping the model in DDP. This is required because:
        # 1. PEFT freezes base model parameters (requires_grad=False)
        # 2. DDP must be aware of which parameters are trainable when building gradient buckets
        # 3. The distributed optimizer must only track trainable (adapter) parameters
        # See Megatron-Bridge docs: training/peft.md

        # Register PEFT transformation as pre-wrap hook if peft_cls is specified
        # This must happen BEFORE DDP wrapping to avoid KeyError with frozen parameters
        if peft_cls is not None:
            from verl.utils.megatron_peft_utils import load_adapter_checkpoint, print_adapter_info

            def peft_pre_wrap_hook(model):
                """Pre-wrap hook that applies PEFT transformation."""
                # Apply PEFT transformation - this will freeze base model and add adapters
                # The PEFT callable handles both freezing and transformation
                transformed_model = peft_cls(model, training=True)

                # Set parameters to save (adapter-only checkpointing)
                peft_cls.set_params_to_save(transformed_model)

                # Load adapter weights if adapter_path is specified
                adapter_path = getattr(peft_config, "adapter_path", None)
                if adapter_path is not None and adapter_path:
                    print(f"Loading adapter weights from: {adapter_path}")
                    load_adapter_checkpoint(transformed_model, adapter_path)

                # Print PEFT statistics
                if torch.distributed.get_rank() == 0:
                    print_adapter_info(transformed_model)

                return transformed_model

            provider.register_pre_wrap_hook(peft_pre_wrap_hook)

        # Register post-creation callbacks (make_value_model, freeze_moe_router) as pre-wrap hooks
        for callback in post_model_creation_callbacks:
            provider.register_pre_wrap_hook(callback)

        # Create DDP config if needed
        ddp_config = None
        if wrap_config.wrap_with_ddp:
            from megatron.bridge.training.config import DistributedDataParallelConfig

            ddp_config_dict = {
                "use_distributed_optimizer": wrap_config.use_distributed_optimizer,
            }
            # Apply any DDP config overrides
            if override_ddp_config is not None:
                ddp_config_dict.update(override_ddp_config)

            ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
            ddp_config.finalize()

        # Now call provide_distributed_model with all hooks registered
        # Hooks will be applied automatically before DDP wrapping
        model = provider.provide_distributed_model(
            wrap_with_ddp=wrap_config.wrap_with_ddp,
            ddp_config=ddp_config,
        )

        # Extract TransformerConfig from the created model
        tf_config = get_model_config(model[0] if isinstance(model, list) else model)
    else:
        model = bridge.get_model(
            post_model_creation_callbacks=post_model_creation_callbacks,
            wrap_with_ddp=wrap_config.wrap_with_ddp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            ddp_config=override_ddp_config,
        )

    return model, tf_config


@torch.no_grad()
def offload_megatron_model_to_cpu(models):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # offload parameters
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to("cpu", non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    gc.collect()
    get_torch_device().empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad and hasattr(buffer, "grad_data_size"):
                        buffer.grad_data.storage().resize_(buffer.grad_data_size)
                        buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
        else:
            # we need this for ref module
            device_id = get_device_id()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
    gc.collect()
    get_torch_device().empty_cache()


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def offload_tensor_to_cpu(tensor):
        if tensor is None:
            return
        tensor.data = tensor.data.to("cpu", non_blocking=True)

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def load_megatron_copy_params(optimizers):
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def load_tensor_to_gpu(tensor):
        if tensor is None:
            return
        device_id = get_device_id()
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        load_tensor_to_gpu(param)
                else:
                    load_tensor_to_gpu(param_group)
        else:
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def offload_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        ## worker may hold zero parameter when enabling custom pipeline layout
        if _opt.optimizer is not None:
            opt_state_dict_values = _opt.optimizer.state.values()
            for v in opt_state_dict_values:
                if "exp_avg" in v:
                    v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
                if "exp_avg_sq" in v:
                    v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)
        gc.collect()
        get_torch_device().empty_cache()


@torch.no_grad()
def load_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        ## worker may hold zero parameter when enabling custom pipeline layout
        if _opt.optimizer is not None:
            # if we are using HybridDeviceOptimizer, we need to only move gpu optimizer state to gpu
            if hasattr(_opt.optimizer, "_move_new_state_to_right_device"):
                _opt.optimizer._move_new_state_to_right_device()
            else:
                opt_state_dict_values = _opt.optimizer.state.values()
                for v in opt_state_dict_values:
                    if "exp_avg" in v:
                        v["exp_avg"] = v["exp_avg"].to(get_device_id(), non_blocking=True)
                    if "exp_avg_sq" in v:
                        v["exp_avg_sq"] = v["exp_avg_sq"].to(get_device_id(), non_blocking=True)
        gc.collect()
        get_torch_device().empty_cache()


def get_dist_checkpoint_path(checkpoint_path):
    local_mkdir_safe(checkpoint_path)
    local_mkdir_safe(os.path.join(checkpoint_path, "dist_ckpt"))
    return os.path.join(checkpoint_path, "dist_ckpt")


def get_hf_model_checkpoint_path(checkpoint_path):
    local_mkdir_safe(checkpoint_path)
    local_mkdir_safe(os.path.join(checkpoint_path, "huggingface"))
    return os.path.join(checkpoint_path, "huggingface")


def get_transformer_config_checkpoint_path(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    return os.path.join(checkpoint_path, "transformer_config.json")


def register_megatron_training_hooks(model: list[torch.nn.Module], optimizer):
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config

    try:
        from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP
    except ImportError:
        megatron_FSDP = DDP

    # register some callbacks for megatron training, following https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.0rc7/megatron/training/training.py#L2039-L2057
    for one_model in model:
        config = get_model_config(one_model)
        config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = finalize_model_grads

        overlap_param_gather = getattr(optimizer.config, "overlap_param_gather", False)
        overlap_grad_reduce = getattr(one_model.ddp_config, "overlap_grad_reduce", False)
        align_grad_reduce = True  # default to True, seldom to be false
        align_param_gather = getattr(one_model.ddp_config, "align_param_gather", False)

        if isinstance(model[0], megatron_FSDP | DDP) and overlap_grad_reduce:
            assert config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
            if len(model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if align_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
                if len(model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if overlap_param_gather and align_param_gather:
            config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
            if len(model) == 1:
                config.param_sync_func = config.param_sync_func[0]


def mapping_string_to_attn_backend(args: dict) -> dict:
    if "attention_backend" in args and isinstance(args["attention_backend"], str):
        from megatron.core.transformer.enums import AttnBackend

        args["attention_backend"] = AttnBackend[args["attention_backend"]]
    return args
