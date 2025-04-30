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

from typing import Any, Callable, Dict, Optional, Type

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
)
from .model_forward import (
    gptmodel_forward,
)
from .model_initializer import (
    BaseModelInitializer,
    DenseModel,
    MixtralModel,
    Qwen2MoEModel,
    Qwen25VLModel,
)
from .weight_converter import (
    McoreToHFWeightConverterDense,
    McoreToHFWeightConverterMixtral,
    McoreToHFWeightConverterQwen2Moe,
)

# Registry for model configuration converters
MODEL_CONFIG_CONVERTER_REGISTRY: Dict[str, Callable[[PretrainedConfig, torch.dtype], TransformerConfig]] = {
    "LlamaForCausalLM": hf_to_mcore_config_dense,
    "Qwen2ForCausalLM": hf_to_mcore_config_dense,
    "Qwen2MoeForCausalLM": hf_to_mcore_config_qwen2moe,
    "DeepseekV3ForCausalLM": hf_to_mcore_config_dpskv3,
    "MixtralForCausalLM": hf_to_mcore_config_mixtral,
    "Qwen2_5_VLForConditionalGeneration": hf_to_mcore_config_qwen2_5_vl,
    "Llama4ForConditionalGeneration": hf_to_mcore_config_llama4,
}

# Registry for model initializers
MODEL_INITIALIZER_REGISTRY: Dict[str, Type[BaseModelInitializer]] = {
    "LlamaForCausalLM": DenseModel,
    "Qwen2ForCausalLM": DenseModel,
    "Qwen2MoeForCausalLM": Qwen2MoEModel,
    "MixtralForCausalLM": MixtralModel,
    "DeepseekV3ForCausalLM": DenseModel,
    "Qwen2_5_VLForConditionalGeneration": Qwen25VLModel,
    "Llama4ForConditionalGeneration": DenseModel,
}

# Registry for model forward functions
MODEL_FORWARD_REGISTRY: Dict[str, Callable] = {
    "LlamaForCausalLM": gptmodel_forward,
    "Qwen2ForCausalLM": gptmodel_forward,
    "Qwen2MoeForCausalLM": gptmodel_forward,
    "MixtralForCausalLM": gptmodel_forward,
    "DeepseekV3ForCausalLM": gptmodel_forward,
    "Qwen2_5_VLForConditionalGeneration": gptmodel_forward,
    "Llama4ForConditionalGeneration": gptmodel_forward,
}

# Registry for model weight converters
MODEL_WEIGHT_CONVERTER_REGISTRY: Dict[str, Type] = {
    "LlamaForCausalLM": McoreToHFWeightConverterDense,
    "Qwen2ForCausalLM": McoreToHFWeightConverterDense,
    "Qwen2MoeForCausalLM": McoreToHFWeightConverterQwen2Moe,
    "MixtralForCausalLM": McoreToHFWeightConverterMixtral,
}


### Only add model registry above and do not change below
def hf_to_mcore_config(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_CONFIG_CONVERTER_REGISTRY:
        raise ValueError(f"Model architectures {arch} converter are not supported for now. Supported architectures: {MODEL_CONFIG_CONVERTER_REGISTRY.keys()}")
    return MODEL_CONFIG_CONVERTER_REGISTRY[arch](hf_config, dtype)


def init_mcore_model(
    tfconfig: TransformerConfig,
    hf_config: PretrainedConfig,
    pre_process: Optional[Callable] = None,
    post_process: Optional[Callable] = None,
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
    arch = hf_config.architectures[0]
    if arch not in MODEL_INITIALIZER_REGISTRY:
        raise ValueError(f"Model architectures {arch} initializer are not supported for now. Supported architectures: {MODEL_INITIALIZER_REGISTRY.keys()}")

    initializer_cls = MODEL_INITIALIZER_REGISTRY[arch]
    initializer = initializer_cls(tfconfig, hf_config)

    return initializer.initialize(pre_process=pre_process, post_process=post_process, share_embeddings_and_output_weights=share_embeddings_and_output_weights, value=value, **extra_kwargs)


def get_mcore_forward_fn(hf_config: PretrainedConfig) -> Callable:
    """
    Get the forward function for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_FORWARD_REGISTRY:
        raise ValueError(f"Model architectures {arch} forward function are not supported for now. Supported architectures: {MODEL_FORWARD_REGISTRY.keys()}")
    return MODEL_FORWARD_REGISTRY[arch]


def get_mcore_weight_converter(hf_config: PretrainedConfig, dtype: torch.dtype) -> Any:
    """
    Get the weight converter for given model architecture.
    """
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_WEIGHT_CONVERTER_REGISTRY:
        raise ValueError(f"Model architectures {arch} weight converter are not supported for now. Supported architectures: {MODEL_WEIGHT_CONVERTER_REGISTRY.keys()}")
    tfconfig = hf_to_mcore_config(hf_config, dtype)
    return MODEL_WEIGHT_CONVERTER_REGISTRY[arch](hf_config, tfconfig)
