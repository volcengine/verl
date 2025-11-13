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
"""PEFT configuration of Megatron for VERL."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).

    Args:
        target_modules: List of module names to apply LoRA to.
            Common targets: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        dim: Rank of the low-rank adaptation matrices.
        alpha: Scaling parameter for LoRA updates.
        dropout: Dropout rate for LoRA layers.
        exclude_modules: List of module names to exclude from LoRA.
        adapter_path: Optional path to pre-trained adapter weights to load.
    """

    target_modules: list[str] = field(default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])
    dim: int = 32
    alpha: float = 16
    dropout: float = 0.0
    exclude_modules: list[str] = field(default_factory=list)
    adapter_path: Optional[str] = None


@dataclass
class CanonicalLoRAConfig:
    """Configuration for Canonical LoRA.

    Canonical LoRA applies separate adapters to individual Q, K, V projections
    and separate adapters to up/gate projections in MLP, unlike performant LoRA
    which uses fused adapters.

    Args:
        target_modules: List of module names to apply canonical LoRA to.
            Available targets: ["linear_q", "linear_k", "linear_v", "linear_proj",
                               "linear_fc1_up", "linear_fc1_gate", "linear_fc2"]
        dim: Rank of the low-rank adaptation matrices.
        alpha: Scaling parameter for LoRA updates.
        dropout: Dropout rate for LoRA layers.
        dropout_position: Where to apply dropout - "pre" or "post".
        lora_A_init_method: Initialization method for LoRA A matrix.
        lora_B_init_method: Initialization method for LoRA B matrix.
        exclude_modules: List of module names to exclude from LoRA.
        adapter_path: Optional path to pre-trained adapter weights to load.
    """

    target_modules: list[str] = field(
        default_factory=lambda: [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]
    )
    dim: int = 32
    alpha: float = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    exclude_modules: list[str] = field(default_factory=list)
    adapter_path: Optional[str] = None


@dataclass
class DoRAConfig:
    """Configuration for DoRA (Weight-Decomposed Low-Rank Adaptation).

    DoRA decomposes pre-trained weights into magnitude and direction, learning
    a separate magnitude parameter while using LoRA for directional updates.

    Args:
        target_modules: List of module names to apply DoRA to.
            Common targets: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        dim: Rank of the low-rank adaptation matrices.
        alpha: Scaling parameter for DoRA updates.
        dropout: Dropout rate for DoRA layers.
        exclude_modules: List of module names to exclude from DoRA.
        adapter_path: Optional path to pre-trained adapter weights to load.
    """

    target_modules: list[str] = field(default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])
    dim: int = 32
    alpha: float = 16
    dropout: float = 0.0
    exclude_modules: list[str] = field(default_factory=list)
    adapter_path: Optional[str] = None


def get_peft_config(model_config, bridge, provider):
    """Get PEFT configuration from model config.

    Args:
        model_config: Model configuration object.
        bridge: Megatron-Bridge AutoBridge instance.
        provider: Provider instance.

    Returns:
        PEFT configuration object (LoRAConfig, CanonicalLoRAConfig, DoRAConfig) or None.
    """
    peft_config = None
    if not hasattr(model_config, "lora"):
        return peft_config

    lora_cfg = model_config.lora
    # Only enable if rank > 0
    if lora_cfg.get("rank", 0) <= 0:
        return peft_config

    assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"

    from verl.workers.config.megatron_peft import CanonicalLoRAConfig, DoRAConfig, LoRAConfig

    lora_type = lora_cfg.get("type", "lora")
    if lora_type == "lora":
        peft_config = LoRAConfig(
            target_modules=lora_cfg.get("target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.0),
            exclude_modules=lora_cfg.get("exclude_modules", []),
            adapter_path=lora_cfg.get("adapter_path", None),
        )
    elif lora_type == "canonical_lora":
        peft_config = CanonicalLoRAConfig(
            target_modules=lora_cfg.get(
                "target_modules",
                [
                    "linear_q",
                    "linear_k",
                    "linear_v",
                    "linear_proj",
                    "linear_fc1_up",
                    "linear_fc1_gate",
                    "linear_fc2",
                ],
            ),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.0),
            exclude_modules=lora_cfg.get("exclude_modules", []),
            adapter_path=lora_cfg.get("adapter_path", None),
        )
    elif lora_type == "dora":
        peft_config = DoRAConfig(
            target_modules=lora_cfg.get("target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]),
            dim=lora_cfg.get("rank"),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.0),
            exclude_modules=lora_cfg.get("exclude_modules", []),
            adapter_path=lora_cfg.get("adapter_path", None),
        )

    print(
        f"Enabling {lora_type.upper()} with rank={lora_cfg.get('rank')}, "
        f"alpha={lora_cfg.get('alpha')}, dropout={lora_cfg.get('dropout')}"
    )
    return peft_config


def peft_config_to_bridge(peft_cfg):
    """Convert VERL PEFT config to Megatron-Bridge PEFT config.

    Args:
        peft_cfg: PEFT configuration (LoRAConfig, CanonicalLoRAConfig, or DoRAConfig)

    Returns:
        Megatron-Bridge PEFT configuration object
    """
    if peft_cfg is None:
        return None

    from verl.models.mcore.bridge import CanonicalLoRA, DoRA, LoRA

    if isinstance(peft_cfg, LoRAConfig):
        return LoRA(
            target_modules=peft_cfg.target_modules,
            dim=peft_cfg.dim,
            alpha=peft_cfg.alpha,
            dropout=peft_cfg.dropout,
            exclude_modules=peft_cfg.exclude_modules,
            # Note: network_alpha is not supported by Megatron-Bridge LoRA for now, although documented
        )
    elif isinstance(peft_cfg, CanonicalLoRAConfig):
        return CanonicalLoRA(
            target_modules=peft_cfg.target_modules,
            dim=peft_cfg.dim,
            alpha=peft_cfg.alpha,
            dropout=peft_cfg.dropout,
            dropout_position=peft_cfg.dropout_position,
            lora_A_init_method=peft_cfg.lora_A_init_method,
            lora_B_init_method=peft_cfg.lora_B_init_method,
            exclude_modules=peft_cfg.exclude_modules,
        )
    elif isinstance(peft_cfg, DoRAConfig):
        return DoRA(
            target_modules=peft_cfg.target_modules,
            dim=peft_cfg.dim,
            alpha=peft_cfg.alpha,
            dropout=peft_cfg.dropout,
            exclude_modules=peft_cfg.exclude_modules,
        )
    else:
        raise ValueError(f"Unknown PEFT config type: {type(peft_cfg)}")


__all__ = [
    "LoRAConfig",
    "CanonicalLoRAConfig",
    "DoRAConfig",
    "get_peft_config",
    "peft_config_to_bridge",
]
