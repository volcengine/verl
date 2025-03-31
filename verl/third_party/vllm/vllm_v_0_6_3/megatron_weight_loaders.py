# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

from typing import Dict

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import *
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models import ModelRegistry


# NOTE(shengguangming): replace the origin weight loader function in the class
def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Parallel Linear weight loader."""
    assert (param.size() == loaded_weight.size(
    )), "the parameter size is not align with the loaded weight size, param size: {}, loaded_weight size: {}".format(
        param.size(), loaded_weight.size())
    assert (param.data.dtype == loaded_weight.data.dtype
           ), "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    assert (param.data.dtype == loaded_weight.data.dtype
           ), "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def gpt2_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            # GPT-2 ties the weights of the embedding layer and the final
            # linear layer.
            continue
        if ".attn.bias" in name or ".attn.masked_bias" in name:
            # Skip attention mask.
            # NOTE: "c_attn.bias" should not be skipped.
            continue
        if not name.startswith("transformer."):
            name = "transformer." + name
        param = params_dict[name]
        # The HF's GPT-2 implementation uses Conv1D instead of Linear.
        # Because of this, we need to transpose the weights.
        # Note(zhuohan): the logic below might break quantized models.
        for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
            if conv1d_weight_name not in name:
                continue
            if not name.endswith(".weight"):
                continue
            # TODO: check megatron
            loaded_weight = loaded_weight.t()
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def llama_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # NOTE(shengguangming): the megatron llama may have this prefix
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def qwen2_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def llama_megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith(".bias") and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        (
            "input_layernorm",
            "input_layernorm",
        ),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        name = _replace_name(name, params_mapping)
        if name.endswith(".bias") and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if "layers" in megatron_name:  # deal with decoder layers
            megatron_name = megatron_name.replace("decoder", "model")
            megatron_name_list = megatron_name.split(".")
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name


def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if "layers" in megatron_name:  # deal with decoder layers
            megatron_name = megatron_name.replace("decoder", "model")
            megatron_name_list = megatron_name.split(".")
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name


def mistral_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # TODO: need to implement a general way to deal with prefix
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    params_dict = dict(vllm_model.named_parameters())
    for original_name, loaded_weight in actor_weights.items():
        name = _replace_name(original_name, params_mapping)
        if not name or name.endswith(".bias") and name not in params_dict:
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


__LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__ = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader,
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): latest commit in vllm fix awq for this function and add load_weights
    # "default_weight_loader": default_weight_loader
}

# for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
#     # setattr(layer_class, 'megatron_weight_loader', weight_loader)
#     layer_class.weight_loader = weight_loader

__MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__ = {
    "GPT2LMHeadModel": gpt2_weight_loader,
    "LlamaForCausalLM": megatron_core_te_weight_loader,  # use te backend for open-source megatron
    "LLaMAForCausalLM": megatron_core_te_weight_loader,
    "MistralForCausalLM": mistral_megatron_weight_loader,
    'Qwen2ForCausalLM': megatron_core_te_weight_loader,
}


# the actor model is .state_dict()
# Load megatron weights
def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def update_megatron_weight_loader():
    for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
        layer_class.weight_loader = weight_loader
