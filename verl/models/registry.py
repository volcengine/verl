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

"""Utilities for resolving Megatron based model classes.

This module maps HuggingFace architecture names to the corresponding
Megatron-Core implementations shipped with :mod:`verl`.  It exposes helper
functions that allow the rest of the code base to dynamically import the
appropriate class given an architecture string.
"""

import importlib
from typing import List, Optional, Type

import torch.nn as nn

# Supported models in Megatron-LM
# Architecture -> (module, class).
_MODELS = {
    "LlamaForCausalLM": (
        "llama",
        ("ParallelLlamaForCausalLMRmPadPP", "ParallelLlamaForValueRmPadPP", "ParallelLlamaForCausalLMRmPad"),
    ),
    "Qwen2ForCausalLM": (
        "qwen2",
        ("ParallelQwen2ForCausalLMRmPadPP", "ParallelQwen2ForValueRmPadPP", "ParallelQwen2ForCausalLMRmPad"),
    ),
    "MistralForCausalLM": (
        "mistral",
        ("ParallelMistralForCausalLMRmPadPP", "ParallelMistralForValueRmPadPP", "ParallelMistralForCausalLMRmPad"),
    ),
}


# return model class
class ModelRegistry:
    """Helper class for obtaining Megatron model implementations."""

    @staticmethod
    def load_model_cls(model_arch: str, value: bool = False) -> Optional[Type[nn.Module]]:
        """Return the Megatron implementation for ``model_arch``.

        Args:
            model_arch: The HuggingFace architecture name.
            value: If ``True`` return the value head implementation.

        Returns:
            The resolved model class or ``None`` if ``model_arch`` is unknown.
        """
        if model_arch not in _MODELS:
            return None

        megatron = "megatron"

        module_name, model_cls_name = _MODELS[model_arch]
        if not value:  # actor/ref
            model_cls_name = model_cls_name[0]
        elif value:  # critic/rm
            model_cls_name = model_cls_name[1]

        module = importlib.import_module(f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        """Return the list of supported architecture names."""
        return list(_MODELS.keys())
