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
"""
Registry module for model architecture components.
"""

from enum import Enum
from typing import Callable, Dict, Type

import torch
import torch.nn as nn

from .config_converter import (
    PretrainedConfig,
    TransformerConfig,
    hf_to_mcore_config_dense,
    hf_to_mcore_config_dpskv3,
    hf_to_mcore_config_llama4,
    hf_to_mcore_config_mixtral,
    hf_to_mcore_config_qwen2_5_vl,
    hf_to_mcore_config_qwen2moe,
    hf_to_mcore_config_qwen3moe,
)
from .model_forward import (
    gptmodel_forward,
)
from .model_initializer import (
    BaseModelInitializer,
    DenseModel,
    MixtralModel,
    Qwen2MoEModel,
    Qwen3MoEModel,
    Qwen25VLModel,
)
from .weight_converter import (
    McoreToHFWeightConverterDense,
    McoreToHFWeightConverterMixtral,
    McoreToHFWeightConverterQwen2Moe,
    McoreToHFWeightConverterQwen3Moe,
)


class SupportedModel(Enum):
    LLAMA = "LlamaForCausalLM"  # tested
    QWEN2 = "Qwen2ForCausalLM"  # tested
    QWEN2_MOE = "Qwen2MoeForCausalLM"  # pending
    DEEPSEEK_V3 = "DeepseekV3ForCausalLM"  # not tested
    MIXTRAL = "MixtralForCausalLM"  # tested
    QWEN2_5_VL = "Qwen2_5_VLForConditionalGeneration"  # not supported
    LLAMA4 = "Llama4ForConditionalGeneration"  # not tested
    QWEN3 = "Qwen3ForCausalLM"  # tested
    QWEN3_MOE = "Qwen3MoeForCausalLM"  # not tested


# Registry for model configuration converters
MODEL_CONFIG_CONVERTER_REGISTRY: Dict[SupportedModel, Callable[[PretrainedConfig, torch.dtype], TransformerConfig]] = {
    SupportedModel.LLAMA: hf_to_mcore_config_dense,
    SupportedModel.QWEN2: hf_to_mcore_config_dense,
    SupportedModel.QWEN2_MOE: hf_to_mcore_config_qwen2moe,
    SupportedModel.DEEPSEEK_V3: hf_to_mcore_config_dpskv3,
    SupportedModel.MIXTRAL: hf_to_mcore_config_mixtral,
    SupportedModel.QWEN2_5_VL: hf_to_mcore_config_qwen2_5_vl,
    SupportedModel.LLAMA4: hf_to_mcore_config_llama4,
    SupportedModel.QWEN3: hf_to_mcore_config_dense,
    SupportedModel.QWEN3_MOE: hf_to_mcore_config_qwen3moe,
}

# Registry for model initializers
MODEL_INITIALIZER_REGISTRY: Dict[SupportedModel, Type[BaseModelInitializer]] = {
    SupportedModel.LLAMA: DenseModel,
    SupportedModel.QWEN2: DenseModel,
    SupportedModel.QWEN2_MOE: Qwen2MoEModel,
    SupportedModel.MIXTRAL: MixtralModel,
    SupportedModel.DEEPSEEK_V3: DenseModel,
    SupportedModel.QWEN2_5_VL: Qwen25VLModel,
    SupportedModel.LLAMA4: DenseModel,
    SupportedModel.QWEN3: DenseModel,
    SupportedModel.QWEN3_MOE: Qwen3MoEModel,
}

# Registry for model forward functions
MODEL_FORWARD_REGISTRY: Dict[SupportedModel, Callable] = {
    SupportedModel.LLAMA: gptmodel_forward,
    SupportedModel.QWEN2: gptmodel_forward,
    SupportedModel.QWEN2_MOE: gptmodel_forward,
    SupportedModel.MIXTRAL: gptmodel_forward,
    SupportedModel.DEEPSEEK_V3: gptmodel_forward,
    SupportedModel.QWEN2_5_VL: gptmodel_forward,
    SupportedModel.LLAMA4: gptmodel_forward,
    SupportedModel.QWEN3: gptmodel_forward,
    SupportedModel.QWEN3_MOE: gptmodel_forward,
}

# Registry for model weight converters
MODEL_WEIGHT_CONVERTER_REGISTRY: Dict[SupportedModel, Type] = {
    SupportedModel.LLAMA: McoreToHFWeightConverterDense,
    SupportedModel.QWEN2: McoreToHFWeightConverterDense,
    SupportedModel.QWEN2_MOE: McoreToHFWeightConverterQwen2Moe,
    SupportedModel.MIXTRAL: McoreToHFWeightConverterMixtral,
    SupportedModel.QWEN3: McoreToHFWeightConverterDense,
    SupportedModel.QWEN3_MOE: McoreToHFWeightConverterQwen3Moe,
}


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(f"Model Type: {model_type} not supported. Supported models: {supported_models}") from err


def hf_to_mcore_config(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    return MODEL_CONFIG_CONVERTER_REGISTRY[model](hf_config, dtype)


def init_mcore_model(
    tfconfig: TransformerConfig,
    hf_config: PretrainedConfig,
    pre_process: bool = True,
    post_process: bool = None,
    *,
    share_embeddings_and_output_weights: bool = False,
    value: bool = False,
    **extra_kwargs,  # may be used for vlm and moe
) -> nn.Module:
    """
    Initialize a Mcore model.

    Args:
        tfconfig: The transformer config.
        hf_config: The HuggingFace config.
        pre_process: Optional pre-processing function.
        post_process: Optional post-processing function.
        share_embeddings_and_output_weights: Whether to share embeddings and output weights.
        value: Whether to use value.
        **extra_kwargs: Additional keyword arguments.

    Returns:
        The initialized model.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    initializer_cls = MODEL_INITIALIZER_REGISTRY[model]
    initializer = initializer_cls(tfconfig, hf_config)
    return initializer.initialize(pre_process=pre_process, post_process=post_process, share_embeddings_and_output_weights=share_embeddings_and_output_weights, value=value, **extra_kwargs)


def get_mcore_forward_fn(hf_config: PretrainedConfig) -> Callable:
    """
    Get the forward function for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    return MODEL_FORWARD_REGISTRY[model]


def get_mcore_weight_converter(hf_config: PretrainedConfig, dtype: torch.dtype) -> Callable:
    """
    Get the weight converter for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    model = get_supported_model(hf_config.architectures[0])
    tfconfig = hf_to_mcore_config(hf_config, dtype)
    return MODEL_WEIGHT_CONVERTER_REGISTRY[model](hf_config, tfconfig)
