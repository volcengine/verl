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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models

from typing import Dict
import torch
import torch.nn as nn


# NOTE(shengguangming): replace the origin weight loader function in the class
def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Parallel Linear weight loader."""
    assert param.size() == loaded_weight.size(
    ), 'the parameter size is not align with the loaded weight size, param size: {}, loaded_weight size: {}'.format(
        param.size(), loaded_weight.size())
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

    param.data = loaded_weight.data


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

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


def llama_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # NOTE(shengguangming): the megatron llama may have this prefix
    prefix = '0.module.module.'
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if name[:len(prefix)] == prefix:
            name = name[len(prefix):]
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def mistral_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # TODO: need to implement a general way to deal with prefix
    prefix = '0.module.module.'
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if name[:len(prefix)] == prefix:
            name = name[len(prefix):]
        if "rotary_emb.inv_freq" in name:
            continue
        else:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
