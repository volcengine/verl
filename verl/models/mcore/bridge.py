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

try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.dora import DoRA
    from megatron.bridge.peft.lora import LoRA
except ImportError:
    # `pip install verl[mcore]` or
    print("Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`")
    raise

import torch
from megatron.core import tensor_parallel


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


class LinearForLastLayer(torch.nn.Linear):
    """
    A custom linear layer implementation for the last layer of a model.

    This layer extends PyTorch's Linear module with functionality specifically designed
    for handling the final layer in transformer models with sequence parallelism.

    Attributes:
        sequence_parallel: Boolean indicating whether sequence parallelism is enabled
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        sequence_parallel: bool,
    ):
        """
        Initializes the LinearForLastLayer.

        Args:
            input_size: The size of the input features
            output_size: The size of the output features
            sequence_parallel (bool): Whether sequence parallelism is enabled
        """
        super().__init__(in_features=input_size, out_features=output_size, bias=False)
        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

    def forward(
        self,
        input_,
    ):
        """
        Forward pass for the linear layer.

        This method computes the linear transformation and handles sequence parallelism
        if enabled, gathering outputs from different sequence parallel regions.

        Args:
            input_: Input tensor

        Returns:
            tuple: (logits, None) where logits is the output of the linear transformation
        """
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


# Make Megatron-Bridge AutoMapping treats the custom last layer as replicated.
AutoMapping.register_module_type("LinearForLastLayer", "replicated")


def make_value_model(hidden_size, sequence_parallel):
    """Creates a pre-wrap hook that prepares (but doesn't activate) a value head.

    This two-phase approach allows HF weight loading to work with the original output layer,
    then activates the value head afterward via `activate_value_head()`.

    Phase 1 (this hook): Stash the original output_layer and create the value head
    Phase 2 (explicit call): Replace output_layer with the value head

    Args:
        hidden_size (int): The hidden size of the model's transformer layers.
        sequence_parallel (bool): Whether sequence parallelism is enabled.

    Returns:
        A hook function that can be used as a `pre_wrap_hook` in Megatron-Bridge.
        The hook itself takes the model as input and prepares it for value head activation.
    """

    def hook(model):
        for model_chunk in _ensure_model_list(model):
            original_layer = getattr(model_chunk, "output_layer", None)
            if original_layer is None:
                continue

            # Skip if already prepared
            if model_chunk.__dict__.get("_value_output_layer") is not None:
                continue

            value_layer = LinearForLastLayer(
                input_size=hidden_size,
                output_size=1,
                sequence_parallel=sequence_parallel,
            )

            # Stash both layers without replacing output_layer yet
            model_chunk.__dict__["_hf_output_layer"] = original_layer
            model_chunk.__dict__["_value_output_layer"] = value_layer

            # Do NOT replace output_layer here - keep original for weight loading

    return hook


def activate_value_head(model) -> bool:
    """Activate the value head after HF weights have been loaded.

    This should be called after `load_hf_weights()` to replace the original
    output_layer with the compact value head prepared by `make_value_model()` hook.

    Args:
        model: Model or list of models with prepared value heads

    Returns:
        bool: True if any output layer was activated
    """
    activated = False
    for model_chunk in _ensure_model_list(model):
        value_layer = model_chunk.__dict__.get("_value_output_layer")
        if value_layer is None:
            continue

        # Only activate if not already active
        if model_chunk.output_layer is not value_layer:
            model_chunk.output_layer = value_layer
            activated = True

    return activated


def deactivate_value_head(model) -> bool:
    """Deactivate the value head before saving HF weights.

    This should be called before `save_hf_weights()` to restore the original
    output_layer so weights are saved in HuggingFace format.

    Args:
        model: Model or list of models with active value heads

    Returns:
        bool: True if any output layer was deactivated
    """
    deactivated = False
    for model_chunk in _ensure_model_list(model):
        orig_layer = model_chunk.__dict__.get("_hf_output_layer")
        if orig_layer is None:
            continue

        # Only deactivate if currently using value head
        value_layer = model_chunk.__dict__.get("_value_output_layer")
        if value_layer is not None and model_chunk.output_layer is value_layer:
            model_chunk.output_layer = orig_layer
            deactivated = True

    return deactivated


def freeze_moe_router(model):
    """Pre-wrap hook to freeze MoE router parameters.

    Args:
        model: List of MegatronModule instances or single module

    Returns:
        The model with frozen router parameters
    """
    for model_chunk in _ensure_model_list(model):
        if hasattr(model_chunk, "decoder") and hasattr(model_chunk.decoder, "layers"):
            for layer in model_chunk.decoder.layers:
                if hasattr(layer.mlp, "router"):
                    if hasattr(layer.mlp.router, "weight"):
                        layer.mlp.router.weight.requires_grad = False
                    if hasattr(layer.mlp.router, "bias"):
                        layer.mlp.router.bias.requires_grad = False
                if hasattr(layer.mlp, "shared_experts"):
                    if hasattr(layer.mlp.shared_experts, "gate_weight"):
                        layer.mlp.shared_experts.gate_weight.requires_grad = False
                    if hasattr(layer.mlp.shared_experts, "gate_bias"):
                        layer.mlp.shared_experts.gate_bias.requires_grad = False

    return model


__all__ = [
    "AutoBridge",
    "make_value_model",
    "freeze_moe_router",
    "LoRA",
    "DoRA",
    "CanonicalLoRA",
    "activate_value_head",
    "deactivate_value_head",
]
