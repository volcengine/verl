# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import torch.nn as nn

from .config_converter import (
    PretrainedConfig,
    TransformerConfig,
    hf_to_mcore_config_dense,
    hf_to_mcore_config_dpskv3,
    hf_to_mcore_config_llama4,
    hf_to_mcore_config_qwen2_5_vl,
    hf_to_mcore_config_qwen2moe,
)
from .model_forward import (
    gptmodel_forward_dense,
    gptmodel_forward_dpskv3,
    gptmodel_forward_llama4,
    gptmodel_forward_qwen2_5_vl,
    gptmodel_forward_qwen2_moe,
)
from .model_initializer import (
    init_mcore_model_dense,
    init_mcore_model_dpskv3,
    init_mcore_model_llama4,
    init_mcore_model_qwen2_5_vl,
    init_mcore_model_qwen2_moe,
)
from .weight_converter import McoreToHFWeightConverterDense, McoreToHFWeightConverterQwen2Moe


def hf_to_mcore_config(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    MODEL_CONFIG_CONVERTER_REGISTRY = {
        "LlamaForCausalLM": hf_to_mcore_config_dense,
        "Qwen2ForCausalLM": hf_to_mcore_config_dense,
        "Qwen2MoeForCausalLM": hf_to_mcore_config_qwen2moe,
        "DeepseekV3ForCausalLM": hf_to_mcore_config_dpskv3,
        "Qwen2_5_VLForConditionalGeneration": hf_to_mcore_config_qwen2_5_vl,
        "Llama4ForConditionalGeneration": hf_to_mcore_config_llama4,
    }
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_CONFIG_CONVERTER_REGISTRY:
        raise ValueError(f"Model architectures {arch} converter are not supported for now. Supported architectures: {MODEL_CONFIG_CONVERTER_REGISTRY.keys()}")
    return MODEL_CONFIG_CONVERTER_REGISTRY[arch](hf_config, dtype)


def init_mcore_model(
    tfconfig,
    hf_config,
    pre_process=None,
    post_process=None,
    share_embeddings_and_output_weights=False,
    value=False,
    **extra_kwargs,  # may be used for vlm and moe
) -> nn.Module:
    MODEL_INITIALIZER_REGISTRY = {
        "LlamaForCausalLM": init_mcore_model_dense,
        "Qwen2ForCausalLM": init_mcore_model_dense,
        "Qwen2MoeForCausalLM": init_mcore_model_qwen2_moe,
        "DeepseekV3ForCausalLM": init_mcore_model_dpskv3,
        "Qwen2_5_VLForConditionalGeneration": init_mcore_model_qwen2_5_vl,
        "Llama4ForConditionalGeneration": init_mcore_model_llama4,
    }
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_INITIALIZER_REGISTRY:
        raise ValueError(f"Model architectures {arch} initializer are not supported for now. Supported architectures: {MODEL_INITIALIZER_REGISTRY.keys()}")
    return MODEL_INITIALIZER_REGISTRY[arch](tfconfig, hf_config, pre_process, post_process, share_embeddings_and_output_weights, value, **extra_kwargs)


def get_mcore_forward_fn(hf_config: PretrainedConfig):
    MODEL_FORWARD_REGISTRY = {
        "LlamaForCausalLM": gptmodel_forward_dense,
        "Qwen2ForCausalLM": gptmodel_forward_dense,
        "Qwen2MoeForCausalLM": gptmodel_forward_qwen2_moe,
        "DeepseekV3ForCausalLM": gptmodel_forward_dpskv3,
        "Qwen2_5_VLForConditionalGeneration": gptmodel_forward_qwen2_5_vl,
        "Llama4ForConditionalGeneration": gptmodel_forward_llama4,
    }
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_FORWARD_REGISTRY:
        raise ValueError(f"Model architectures {arch} forward function are not supported for now. Supported architectures: {MODEL_FORWARD_REGISTRY.keys()}")
    return MODEL_FORWARD_REGISTRY[arch]


def get_mcore_weight_converter(hf_config: PretrainedConfig, dtype: torch.dtype):
    MODEL_WEIGHT_CONVERTER_REGISTRY = {
        "LlamaForCausalLM": McoreToHFWeightConverterDense,
        "Qwen2ForCausalLM": McoreToHFWeightConverterDense,
        "Qwen2MoeForCausalLM": McoreToHFWeightConverterQwen2Moe,
    }
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_WEIGHT_CONVERTER_REGISTRY:
        raise ValueError(f"Model architectures {arch} weight converter are not supported for now. Supported architectures: {MODEL_WEIGHT_CONVERTER_REGISTRY.keys()}")
    tfconfig = hf_to_mcore_config(hf_config, dtype)
    return MODEL_WEIGHT_CONVERTER_REGISTRY[arch](hf_config, tfconfig)
