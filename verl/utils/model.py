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
Utilities to create common models from huggingface
"""
import os
import warnings
from typing import Dict, Type, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, MistralForSequenceClassification, GenerationConfig
from verl.models.registry import ModelRegistry


class LambdaLayer(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def update_model_config(module_config, override_config_kwargs):
    for key, val in override_config_kwargs.items():
        setattr(module_config, key, val)


def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> Dict:
    if override_config_kwargs is None:
        override_config_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    update_model_config(module_config, override_config_kwargs)

    return module_config


def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional[GenerationConfig]:
    try:
        return GenerationConfig.from_pretrained(model)
    except OSError:  # Not found
        try:
            config = get_huggingface_actor_config(
                model,
                trust_remote_code=trust_remote_code,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    if override_config_kwargs is None:
        override_config_kwargs = {}
    if automodel_kwargs is None:
        automodel_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    module_config = get_huggingface_actor_config(model_name,
                                                 override_config_kwargs,
                                                 trust_remote_code=automodel_kwargs.get('trust_remote_code', False))
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)
    return module


def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """

    Args:
        model_name:
        override_config_kwargs:

    Returns:

    """
    critic_module: nn.Module = create_huggingface_actor(model_name,
                                                        override_config_kwargs=override_config_kwargs,
                                                        automodel_kwargs=automodel_kwargs)
    if automodel_kwargs is None:
        automodel_kwargs = {}
    torch_dtype = automodel_kwargs.get('torch_dtype', torch.float32)
    critic_module.lm_head = nn.Sequential(nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype),
                                          LambdaLayer(fn=squeeze))
    return critic_module


def get_model_size(model: nn.Module, scale='auto'):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == 'auto':
        if n_params > 1e9:
            scale = 'B'
        elif n_params > 1e6:
            scale = 'M'
        elif n_params > 1e3:
            scale = 'K'
        else:
            scale = ''

    if scale == 'B':
        n_params = n_params / 1e9
    elif scale == 'M':
        n_params = n_params / 1e6
    elif scale == 'K':
        n_params = n_params / 1e3
    elif scale == '':
        pass
    else:
        raise NotImplemented(f'Unknown scale {scale}')

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale='auto')
    if name is None:
        name = model.__class__.__name__
    print(f'{name} contains {n_params:.2f}{scale} parameters')


def create_random_mask(input_ids: torch.Tensor,
                       max_ratio_of_valid_token: float,
                       max_ratio_of_left_padding: float,
                       min_ratio_of_valid_token: float = 0):
    """Create a random mask given input_ids. Support left padding and right padding.
    Process:
    - Sample valid token length
    - Sample left_padding length
    - Generate padding

    Args:
        input_ids:
            shape (batch_size, seq_len)

    Returns:

    """
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token

    batch_size, sequence_length = input_ids.shape
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length
    masks = torch.ones_like(input_ids, dtype=torch.int64)
    # TODO: we can make this faster
    for i in range(batch_size):
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)

        for index in range(num_left_padding):
            masks[i, index] = 0

        for index in range(num_left_padding + num_valid, sequence_length):
            masks[i, index] = 0
    return masks


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def normalize_pp_vpp_params(params, num_hidden_layers, layer_name='layers'):
    """
    Normalize the pp vpp params into a complete named parameters.
    This is useful when gather parameters from pp ranks and passed to a model without pp

    params: List[List[Dict[str, param]]]
        params contains a list of pp, with a list of vpp named_parameters in each vpp chunk.
    output: Dict[str, param]

    """

    def normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_layers):
        """
        Transform the model name in each model_chunk in each pp stage into the name in inference engine
        """
        if vpp_size > 1:
            # print(f'try to bind vpp params to inference engine...')
            layers_per_pp = num_layers // pp_size
            layers_per_vpp = layers_per_pp // vpp_size
            pp_offset = layers_per_vpp * pp_rank
            vpp_offset = (layers_per_vpp * pp_size) * vpp_rank
            layer_offset = pp_offset + vpp_offset
        else:
            layers_per_pp = num_layers // pp_size
            layer_offset = layers_per_pp * pp_rank

        if layer_name in name:  # belong to an intermediate layer
            split_name = name.split('.')
            # find the num next to split_name
            for i, name in enumerate(split_name):
                if name == layer_name:
                    break
            layer_num_idx = i + 1
            # check the name
            assert len(split_name) >= layer_num_idx + 1, f'split_name = {split_name}'
            assert split_name[layer_num_idx].isdigit(), f'split_name = {split_name}'
            # increment layer_num_idx by layer_offset
            split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
            name = '.'.join(split_name)  # weight name in inference_tp_model
        return name

    pp_size = len(params)
    normalized_name_to_param = {}
    for pp_rank in range(len(params)):
        vpp_size = len(params[pp_rank])
        for vpp_rank in range(vpp_size):
            for name, param in params[pp_rank][vpp_rank].items():
                normalized_name = normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_hidden_layers)
                normalized_name_to_param[normalized_name] = param

    return normalized_name_to_param


def get_parallel_model_from_config(config,
                                   megatron_config,
                                   pre_process=None,
                                   post_process=None,
                                   share_embeddings_and_output_weights=False,
                                   value=False):
    from megatron.core import ModelParallelConfig
    assert isinstance(megatron_config, ModelParallelConfig)
    model_class = _get_parallel_model_architecture_from_config(config, value)

    model = model_class(config,
                        megatron_config,
                        pre_process=pre_process,
                        post_process=post_process,
                        share_embeddings_and_output_weights=share_embeddings_and_output_weights)
    return model


def _get_parallel_model_architecture_from_config(config: PretrainedConfig, value=False) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch, value)
        if model_cls is not None:
            return model_cls
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def load_megatron_model_weights(config,
                                model_config,
                                parallel_model,
                                params_dtype,
                                is_value_model=False,
                                local_cache_path='~/.cache/verl/rlhf'):
    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    architectures = getattr(model_config, "architectures", [])
    local_cache_path = os.path.expanduser(local_cache_path)

    if config.model.path.startswith("hdfs:"):
        from verl.utils.fs import copy_to_local
        print(f'start download from {config.model.path}')
        local_model_path = copy_to_local(src=config.model.path, cache_dir=local_cache_path)
        print('finish download')
    else:
        print(f"load from local dir {config.model.path}")
        local_model_path = config.model.path

    # TODO: to find a better way to load mistral7b-rm lm_head
    from verl.utils.fsdp_utils import get_init_weight_context_manager
    init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if 'mistral7b-rm' in config.model.path:
            model = MistralForSequenceClassification.from_pretrained(
                local_model_path)  # use score head instead of lm_head
            state_dict = model.state_dict()
            state_dict['lm_head.weight'] = state_dict['score.weight']
            state_dict['model.embed_tokens.weight'] = state_dict[
                'model.embed_tokens.weight'][:32000]  # workaround, 32001 -> 32000
            is_value_model = True
        else:
            model = AutoModelForCausalLM.from_pretrained(local_model_path)
            state_dict = model.state_dict()

    from verl.models.weight_loader_registry import get_weight_loader
    print(f'before weight loader: architectures = {architectures}...')
    for arch in architectures:
        print(f'call weight loader arch = {arch}, model config = {model.config}')
        weight_loader = get_weight_loader(arch)
        weight_loader(state_dict=state_dict,
                      wrapped_models=parallel_model,
                      config=model.config,
                      params_dtype=params_dtype,
                      is_value_model=is_value_model,
                      tie_word_embeddings=model_config.tie_word_embeddings)
    return model.config


# pad input_ids_rmpad, cu_seqlens and max_seqlen_in_batch to be divisible by tp
def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    """pad the tokens such that the total length is a multiple of size.
    This function is useful when applying sequence parallel and context parallel

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding
        cu_seqlens: (total_nnz + 1,)
        max_seqlen_in_batch: int

    Returns:

    """
    F = nn.functional

    total_nnz = unpad_tokens.shape[0]

    if total_nnz % size == 0:
        pad_size = 0
    else:
        pad_size = size - total_nnz % size

    # we assume adding a new data in the batch with seqlen pad_size
    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(f'Padding dim {unpad_tokens.ndim()} is not supported')

        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)

    return unpad_tokens, cu_seqlens, max_seqlen_in_batch
