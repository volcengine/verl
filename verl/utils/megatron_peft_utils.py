# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Utilities for PEFT (Parameter-Efficient Fine-Tuning) of Megatron in VERL."""

import os
from pathlib import Path

import torch


def _get_rank_checkpoint_path(base_path: str) -> str:
    """Get rank-specific checkpoint path following Megatron's convention.

    Returns path like: base_path/mp_rank_{tp:02d}_{pp:03d}_{ep:03d}/

    Args:
        base_path: Base checkpoint directory

    Returns:
        Rank-specific subdirectory path
    """
    from megatron.core import mpu

    tensor_rank = mpu.get_tensor_model_parallel_rank()
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    expert_rank = mpu.get_expert_model_parallel_rank()

    pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    expert_parallel = mpu.get_expert_model_parallel_world_size() > 1

    if not pipeline_parallel:
        rank_path = os.path.join(base_path, f"mp_rank_{tensor_rank:02d}")
    else:
        rank_path = os.path.join(base_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

    if expert_parallel:
        rank_path = rank_path + f"_{expert_rank:03d}"

    return rank_path


def get_adapter_state_dict(model):
    """Extract only adapter parameters from a model.

    Args:
        model: PyTorch model (possibly wrapped in DDP/Float16Module)

    Returns:
        Dict of adapter parameter names to tensors
    """
    from verl.utils.megatron_utils import unwrap_model

    # Unwrap model from DDP/Float16Module
    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        unwrapped = unwrapped[0]

    adapter_state = {}
    for name, param in unwrapped.named_parameters():
        if ".adapter." in name.lower():
            adapter_state[name] = param.data.clone()

    return adapter_state


def save_adapter_checkpoint(
    model: torch.nn.Module | list[torch.nn.Module],
    checkpoint_path: str,
    rank: int = 0,
):
    """Save only adapter parameters to checkpoint.

    This is much more efficient than saving the full model when using PEFT,
    as adapters typically represent <1% of total parameters.

    Uses Megatron's distributed checkpoint structure: each rank saves to
    checkpoint_path/mp_rank_{tp:02d}_{pp:03d}/adapter.pt

    Args:
        model: Model or list of models
        checkpoint_path: Base path to save checkpoint (rank-specific subdirs created)
        rank: Process rank (used for logging only)
    """

    if isinstance(model, list):
        models = model
    else:
        models = [model]

    # Get adapter state from first model
    adapter_state = get_adapter_state_dict(models[0])

    if not adapter_state:
        if rank == 0:
            print("Warning: No adapter parameters found to save")
        return

    # Get rank-specific directory path
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    rank_path = _get_rank_checkpoint_path(checkpoint_path)
    adapter_file = rank_path + "_adapter.pt"

    torch.save(
        {
            "adapter_state_dict": adapter_state,
        },
        adapter_file,
    )

    if rank == 0:
        print(f"Saved {len(adapter_state)} adapter parameters to {checkpoint_path} (distributed)")


def load_adapter_checkpoint(
    model: torch.nn.Module | list[torch.nn.Module],
    checkpoint_path: str,
    strict: bool = True,
):
    """Load adapter parameters from checkpoint.

    Loads from Megatron's distributed checkpoint structure: reads from
    checkpoint_path/mp_rank_{tp:02d}_{pp:03d}/adapter.pt for each rank.

    Args:
        model: Model or list of models
        checkpoint_path: Base path to checkpoint directory
        strict: Whether to strictly enforce parameter name matching
    """
    from megatron.core import mpu

    from verl.utils.megatron_utils import unwrap_model

    # Get rank-specific path
    rank_path = _get_rank_checkpoint_path(checkpoint_path)
    adapter_file = rank_path + "_adapter.pt"

    if not os.path.isfile(adapter_file):
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_file}")

    checkpoint = torch.load(adapter_file, map_location="cpu")
    adapter_state = checkpoint.get("adapter_state_dict", {})

    if not adapter_state:
        print("Warning: No adapter parameters found in checkpoint")
        return

    if isinstance(model, list):
        models = model
    else:
        models = [model]

    # Load adapter parameters into each model (for VPP, models may have multiple chunks)
    loaded_count = 0
    for m in models:
        unwrapped = unwrap_model(m)
        if isinstance(unwrapped, list):
            unwrapped = unwrapped[0]

        # Load parameters
        _, unexpected = unwrapped.load_state_dict(adapter_state, strict=False)

        if strict and unexpected:
            raise RuntimeError(f"Error loading adapter checkpoint:\nUnexpected keys: {unexpected}")

        loaded_count += len(adapter_state)

    if (
        mpu.get_data_parallel_rank() == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
    ):
        print(f"Loaded {len(adapter_state)} adapter parameters from {checkpoint_path}")


def count_adapter_parameters(model):
    """Count the number of trainable adapter parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (adapter_params, total_params, percentage)
    """
    from verl.utils.megatron_utils import unwrap_model

    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        unwrapped = unwrapped[0]

    adapter_params = 0
    total_params = 0

    for name, param in unwrapped.named_parameters():
        total_params += param.numel()
        if "lora" in name.lower() or "adapter" in name.lower():
            if param.requires_grad:
                adapter_params += param.numel()

    percentage = 100 * adapter_params / total_params if total_params > 0 else 0

    return adapter_params, total_params, percentage


def print_adapter_info(model):
    """Print information about adapter parameters in the model."""
    adapter_params, total_params, percentage = count_adapter_parameters(model)

    print(f"\n{'=' * 60}")
    print("PEFT Adapter Information:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Adapter parameters:   {adapter_params:,}")
    print(f"  Trainable percentage: {percentage:.2f}%")
    print(f"{'=' * 60}\n")


__all__ = [
    "get_adapter_state_dict",
    "save_adapter_checkpoint",
    "load_adapter_checkpoint",
    "count_adapter_parameters",
    "print_adapter_info",
]
