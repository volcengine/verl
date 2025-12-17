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
FSDP2-compatible TiledMLP for memory-efficient MLP computation.

This module patches MLP classes to use tiled computation, reducing peak memory
by processing input in chunks while maintaining gradient correctness.

Usage (must be called BEFORE model instantiation):
    from verl.utils.experimental.tiled_mlp import patch_mlp_for_tiling

    # Patch before loading model
    patch_mlp_for_tiling(num_tiles=4)

    # Then load model
    model = AutoModelForCausalLM.from_pretrained(...)
"""

import threading
from typing import Callable, Dict, List, Set, Type

import torch
import torch.nn as nn

# Global config
_TILED_MLP_CONFIG: Dict[str, int] = {"num_tiles": 0}
_PATCHED_CLASSES: Set[Type] = set()


class _GradientAccumulator:
    """Accumulates gradients across tiles for FSDP compatibility."""

    __slots__ = ("params", "num_tiles", "dtype", "accumulated", "hooks", "lock")

    def __init__(self, params: List[nn.Parameter], num_tiles: int, dtype: torch.dtype):
        self.params = params
        self.num_tiles = num_tiles
        self.dtype = dtype
        self.accumulated = {p: torch.zeros_like(p, dtype=dtype) for p in params}
        self.hooks: List = []
        self.lock = threading.Lock()

    def install_hooks(self, is_last_tile: bool):
        self._remove_hooks()

        def make_hook(param):
            def hook(grad):
                with self.lock:
                    self.accumulated[param].add_(grad.to(self.dtype))
                    if is_last_tile:
                        return self.accumulated[param].to(param.dtype)
                    return None

            return hook

        for p in self.params:
            if p.requires_grad:
                self.hooks.append(p.register_hook(make_hook(p)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def cleanup(self):
        self._remove_hooks()


class _TiledMLPFunction(torch.autograd.Function):
    """Autograd function for tiled MLP computation."""

    @staticmethod
    def forward(ctx, mlp_fn: Callable, module: nn.Module, x: torch.Tensor, num_tiles: int, *params):
        ctx.mlp_fn = mlp_fn
        ctx.module = module
        ctx.num_tiles = num_tiles
        ctx.params = [p for p in params if p.requires_grad]
        ctx.save_for_backward(x)

        # Forward: tile along sequence dimension
        tiles = list(torch.chunk(x, num_tiles, dim=-2))
        with torch.no_grad():
            outputs = [mlp_fn(module, tile) for tile in tiles]
        return torch.cat(outputs, dim=-2)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mlp_fn, module, num_tiles = ctx.mlp_fn, ctx.module, ctx.num_tiles
        params = ctx.params

        x_requires_grad = x.requires_grad
        x = x.detach().requires_grad_(x_requires_grad)

        # Flatten to [N, hidden_size] for tiling
        hidden_size = x.shape[-1]
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_size)
        grad_flat = grad_output.view(-1, hidden_size)

        # Pre-allocate input gradient
        x_grad = torch.zeros_like(x_flat) if x_requires_grad else None

        tiles = list(torch.chunk(x_flat, num_tiles, dim=0))
        accumulator = _GradientAccumulator(params, num_tiles, x.dtype)

        for i, tile in enumerate(tiles):
            tile = tile.detach().requires_grad_(x_requires_grad)
            tile_size = tile.shape[0]
            offset = i * tiles[0].shape[0]

            if x_requires_grad:
                tile.grad = x_grad.narrow(0, offset, tile_size)

            grad_tile = grad_flat.narrow(0, offset, tile_size)
            accumulator.install_hooks(is_last_tile=(i == num_tiles - 1))

            with torch.enable_grad():
                out = mlp_fn(module, tile)
            torch.autograd.backward(out, grad_tile)

        accumulator.cleanup()

        x_grad = x_grad.view(orig_shape) if x_requires_grad else None
        return (None, None, x_grad, None) + (None,) * len(params)


def tiled_mlp_forward(
    mlp_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    module: nn.Module,
    x: torch.Tensor,
    params: List[nn.Parameter],
    num_tiles: int = 4,
) -> torch.Tensor:
    """
    Execute MLP forward with tiled computation for memory efficiency.

    Args:
        mlp_fn: Function that computes MLP output given (module, input).
        module: The MLP module instance.
        x: Input tensor of shape [..., seq_len, hidden_size].
        params: List of parameters that require gradient accumulation.
        num_tiles: Number of tiles to split the input into.

    Returns:
        Output tensor with same shape as input (except last dim may differ).
    """
    if num_tiles <= 1 or not x.requires_grad:
        return mlp_fn(module, x)

    return _TiledMLPFunction.apply(mlp_fn, module, x, num_tiles, *params)


# =============================================================================
# MLP Patching Functions
# =============================================================================


def _create_tiled_forward(original_forward: Callable, get_params_fn: Callable) -> Callable:
    """Create a tiled forward function that wraps the original."""

    def tiled_forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tiles = _TILED_MLP_CONFIG.get("num_tiles", 0)
        if num_tiles <= 1 or not x.requires_grad:
            return original_forward(self, x)

        params = get_params_fn(self)
        return tiled_mlp_forward(original_forward, self, x, params, num_tiles)

    return tiled_forward


def _get_llama_mlp_params(mlp) -> List[nn.Parameter]:
    """Get parameters for LlamaMLP-style architectures."""
    params = []
    for name in ["gate_proj", "up_proj", "down_proj"]:
        if hasattr(mlp, name):
            proj = getattr(mlp, name)
            if hasattr(proj, "weight"):
                params.append(proj.weight)
    return params


def _get_gpt2_mlp_params(mlp) -> List[nn.Parameter]:
    """Get parameters for GPT2MLP-style architectures."""
    params = []
    for name in ["c_fc", "c_proj"]:
        if hasattr(mlp, name):
            proj = getattr(mlp, name)
            if hasattr(proj, "weight"):
                params.append(proj.weight)
    return params


def patch_mlp_for_tiling(num_tiles: int = 4) -> None:
    """
    Patch MLP classes to use tiled computation. MUST be called BEFORE model instantiation.

    This function patches the forward method of common MLP classes (LlamaMLP, Qwen2MLP, etc.)
    to use memory-efficient tiled computation that is compatible with FSDP2.

    Args:
        num_tiles: Number of tiles to split the sequence into. Set to 0 or 1 to disable.

    Example:
        from verl.utils.experimental.tiled_mlp import patch_mlp_for_tiling

        # Enable tiled MLP before loading model
        patch_mlp_for_tiling(num_tiles=4)

        # Load model - MLP will automatically use tiled computation
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    """
    global _TILED_MLP_CONFIG

    _TILED_MLP_CONFIG["num_tiles"] = num_tiles

    if num_tiles <= 1:
        print(f"TiledMLP disabled (num_tiles={num_tiles})")
        return

    # Patch LlamaMLP-style classes (Llama, Qwen2, Mistral, etc.)
    _patch_llama_style_mlp()

    print(f"TiledMLP enabled with {num_tiles} tiles")


def _patch_llama_style_mlp() -> None:
    """Patch LlamaMLP and similar architectures."""
    mlp_classes_to_patch = []

    # Try to import various MLP classes
    try:
        from transformers.models.llama.modeling_llama import LlamaMLP

        mlp_classes_to_patch.append(("LlamaMLP", LlamaMLP, _get_llama_mlp_params))
    except ImportError:
        pass

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        mlp_classes_to_patch.append(("Qwen2MLP", Qwen2MLP, _get_llama_mlp_params))
    except ImportError:
        pass

    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

        mlp_classes_to_patch.append(("Qwen3MLP", Qwen3MLP, _get_llama_mlp_params))
    except ImportError:
        pass

    try:
        from transformers.models.mistral.modeling_mistral import MistralMLP

        mlp_classes_to_patch.append(("MistralMLP", MistralMLP, _get_llama_mlp_params))
    except ImportError:
        pass

    try:
        from transformers.models.gemma.modeling_gemma import GemmaMLP

        mlp_classes_to_patch.append(("GemmaMLP", GemmaMLP, _get_llama_mlp_params))
    except ImportError:
        pass

    try:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP

        mlp_classes_to_patch.append(("Gemma2MLP", Gemma2MLP, _get_llama_mlp_params))
    except ImportError:
        pass

    # Apply patches
    for name, mlp_class, get_params_fn in mlp_classes_to_patch:
        if mlp_class in _PATCHED_CLASSES:
            continue

        original_forward = mlp_class.forward
        mlp_class.forward = _create_tiled_forward(original_forward, get_params_fn)
        _PATCHED_CLASSES.add(mlp_class)
        print(f"  Patched {name}.forward for tiled computation")


def unpatch_mlp() -> None:
    """
    Disable tiled MLP computation.

    Note: This only disables the tiling behavior by setting num_tiles to 0.
    The patched forward methods remain but will fall back to original behavior.
    """
    global _TILED_MLP_CONFIG
    _TILED_MLP_CONFIG["num_tiles"] = 0
    print("TiledMLP disabled")


def get_tiled_mlp_config() -> Dict[str, int]:
    """Get current TiledMLP configuration."""
    return _TILED_MLP_CONFIG.copy()


def patch_mlp_for_model(model: nn.Module, num_tiles: int = 4) -> None:
    """
    Patch MLP modules in an already instantiated model for tiled computation.

    This function finds all MLP modules in the model and patches their forward
    methods to use memory-efficient tiled computation.

    Args:
        model: The instantiated model to patch.
        num_tiles: Number of tiles to split the sequence into.

    Example:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        patch_mlp_for_model(model, num_tiles=4)
    """
    if num_tiles <= 1:
        return

    # Update global config
    global _TILED_MLP_CONFIG
    _TILED_MLP_CONFIG["num_tiles"] = num_tiles

    # Find and patch MLP modules
    patched_count = 0
    for name, module in model.named_modules():
        mlp_class = type(module)
        class_name = mlp_class.__name__

        # Check if this is an MLP-like module
        if not _is_mlp_module(module):
            continue

        # Skip if already patched
        if mlp_class in _PATCHED_CLASSES:
            continue

        # Determine parameter getter
        get_params_fn = _get_params_fn_for_module(module)
        if get_params_fn is None:
            continue

        # Patch the class
        original_forward = mlp_class.forward
        mlp_class.forward = _create_tiled_forward(original_forward, get_params_fn)
        _PATCHED_CLASSES.add(mlp_class)
        patched_count += 1

    if patched_count > 0:
        print(f"TiledMLP: Patched {patched_count} MLP class(es) with {num_tiles} tiles")


def _is_mlp_module(module: nn.Module) -> bool:
    """Check if a module is an MLP-like module."""
    class_name = type(module).__name__.lower()

    # Check by class name
    if "mlp" in class_name:
        return True

    # Check by having typical MLP attributes
    has_gate_up_down = all(hasattr(module, attr) for attr in ["gate_proj", "up_proj", "down_proj"])
    has_fc = all(hasattr(module, attr) for attr in ["c_fc", "c_proj"])

    return has_gate_up_down or has_fc


def _get_params_fn_for_module(module: nn.Module) -> Callable:
    """Get the appropriate parameter getter function for a module."""
    # LlamaMLP-style (gate_proj, up_proj, down_proj)
    if all(hasattr(module, attr) for attr in ["gate_proj", "up_proj", "down_proj"]):
        return _get_llama_mlp_params

    # GPT2-style (c_fc, c_proj)
    if all(hasattr(module, attr) for attr in ["c_fc", "c_proj"]):
        return _get_gpt2_mlp_params

    return None
