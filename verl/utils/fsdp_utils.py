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

from typing import Dict
import functools
import json
import math
import itertools
import os
from contextlib import contextmanager
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from transformers.trainer_pt_utils import get_module_class_from_name
import torch
import torch.nn as nn
import torch.distributed as dist


def init_fn(x: torch.nn.Module):
    if not torch.distributed.get_rank() == 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True):
    from accelerate import init_empty_weights
    cpu_init_weights = lambda: torch.device('cpu')
    if use_meta_tensor:
        init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.
    
    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            if (len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and
                    module.weight.requires_grad):
                return True
            return False

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, f"Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert flat_param.data.data_ptr() == flat_param._local_shard.data_ptr() and \
            id(flat_param.data) != id(flat_param._local_shard) and \
            flat_param.data.size() == flat_param._local_shard.size()
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, f"Only support root model loading to GPU"
    device_id = torch.cuda.current_device()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


@contextmanager
def meta_device_init():
    """
    Create model parameters with meta device.

    Note buffers in model will still be initialized in default device (e.g., CPU),
    since the buffers can be non-persistent and filled with expected values that can
    NOT be captured in meta device.
    """
    device = torch.device("meta")
    old_register_parameter = nn.Module.register_parameter
    registered = set()

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        # we will skip register shared parameters as it
        # is already registered previously
        if param is not None and param not in registered:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            registered.add(module._parameters[name])

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        registered.clear()
        nn.Module.register_parameter = old_register_parameter


def parallel_load_safetensors(filepath):
    """
    Parallel load safetensors from huggingface checkpoint

    Huggingface checkpoint contains:

    - config.json: a json file for model configuration
    - model.safetensor.index.json: a json file for safetensors (parameters & buffers) index
    - model-000x-of-ooxx.safetensors: a binary file for safetensors (parameters & buffers) chunks

    Or (when model is small),

    - model.safetensors: a binary file for all parameters and buffers

    Each rank will own a part of model chunks and load them directly into GPU memory.
    """
    from safetensors.torch import load_file

    safetensors2param = {}

    index_file = os.path.join(filepath, "model.safetensors.index.json")
    if os.path.exists(index_file):
        index = json.load(open(index_file, "rb"))
        for param_name, filename in index["weight_map"].items():
            safetensors2param.setdefault(filename, []).append(param_name)
    else:
        # in this case, the model is small and we can load it all at once
        param_file = os.path.join(filepath, "model.safetensors")
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        states = load_file(param_file)
        for param_name in states:
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        del states

    total_files = len(safetensors2param)
    ckpt_chunks = sorted(safetensors2param.keys())
    world_size = dist.get_world_size()
    size = int(math.ceil(total_files / world_size))
    ckpt_chunks = [ckpt_chunks[rank * size:rank * size + size] for rank in range(world_size)]

    shard_states = {}
    device = torch.cuda.current_device()
    for rank, files in enumerate(ckpt_chunks):
        if rank == dist.get_rank():
            for file in files:
                file = os.path.join(filepath, file)
                states = load_file(file, device=device)
                # print(f"rank {rank} loading {file}...")
                shard_states.update(states)
        else:
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states


def parallel_init_module_fn(module: torch.nn.Module, shard_states: Dict[str, torch.nn.Parameter]):
    """
    Generate a function to initialize sub-modules in the `module` with `shard_states`
    from huggingface checkpoint.

    Args:
        module (torch.nn.Module): the global module to be initialized
        shard_states (Dict[str, torch.nn.Parameter]): the shard states from huggingface checkpoint

    Returns:
        init_fn (Callable): a function to initialize sub-modules in the `module` with `shard_states`
    """

    state2fqn = {}
    for name, state in itertools.chain(module.named_parameters(remove_duplicate=False),
                                       module.named_buffers(remove_duplicate=False)):
        state2fqn.setdefault(state, []).append(name)
    # remove standalone parameters and buffers
    shared = {s for s, names in state2fqn.items() if len(names) > 1}
    materialized_states = {}

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        assert param_name in shard_states, f"{param_name} not loaded"
        device = torch.cuda.current_device()
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:  # buffer
            param = torch.empty_like(state.data, device=device)
        loaded = shard_states[param_name]
        if isinstance(loaded, (torch.nn.Parameter, torch.Tensor)):
            # NOTE: loaded.dtype can be different with param.dtype
            param.data.copy_(loaded.data)
            dist.broadcast(param.data, src=dist.get_rank())
        else:
            assert isinstance(loaded, int)  # the rank that holds the state
            dist.broadcast(param.data, src=loaded)
        shard_states.pop(param_name)
        del loaded
        return param

    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        for name, state in param_and_buffers:
            if not state.is_meta:
                continue
            is_param = name in sub_mod._parameters
            fqn = state2fqn[state].pop(0)
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. "
                        "Such buffer is not saved in checkpoint and user should guarantee to init in CPU / GPU device.")
                continue
            # for shared parameter, we get it from the first time it is created
            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]
            # for not shared parameter, we create it directly
            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state
            else:
                sub_mod._buffers[name] = materialize_state
        if recurse:
            for module in sub_mod.children():
                init_fn(module, recurse=True)

        # for debug
        # if len(shard_states) == 0: print("clear")
        return sub_mod

    return init_fn
