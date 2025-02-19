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

from typing import Dict, Union, Optional, Iterable, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def update_hf_weight_loader():
    from vllm.model_executor.models.gemma import GemmaForCausalLM
    GemmaForCausalLM.load_weights = gemma_load_weights


def gemma_load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())
    loaded_params = set()
    for name, loaded_weight in weights:
        for (param_name, shard_name, shard_id) in stacked_params_mapping:
            if shard_name not in name:
                continue
            name = name.replace(shard_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # lm_head is not used in vllm as it is tied with embed_token.
            # To prevent errors, skip loading lm_head.weight.
            if "lm_head.weight" in name:
                continue
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            # GemmaRMSNorm is different from Llama's in that it multiplies
            # (1 + weight) to the output, instead of just weight.
            if "norm.weight" in name:
                norm_weight = loaded_weight + 1.0  # prevent inplace modify actor weights
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, norm_weight)
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    unloaded_params = params_dict.keys() - loaded_params
    if unloaded_params:
        raise RuntimeError("Some weights are not initialized from checkpoints: "
                           f"{unloaded_params}")


def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    assert isinstance(actor_weights, Dict)
    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):  # TODO
        vllm_model.load_weights(actor_weights.items())
    for _, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
        # FIXME: Remove this after Mixtral is updated
        # to use quant_method.
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
    vllm_model = vllm_model.cuda()
