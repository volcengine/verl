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
Speculator adapter interface and factory.
This keeps trainers decoupled from concrete speculator implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any, Optional

from verl.utils.import_utils import load_class_from_fqn, load_extern_object
from verl.utils.checkpoint.fsdp_checkpoint_manager import load_speculator_checkpoint, save_speculator_checkpoint
from verl.utils.fsdp_utils import fully_shard, maybe_patch_fsdp_module
import torch


class SpeculatorAdapter(ABC):
    has_speculator: bool
    device_mesh: Any
    speculator: Any

    @abstractmethod
    def build_and_attach(self, model, attach_to_model: bool = True):
        """Build and optionally attach a speculator module to the model."""

    @abstractmethod
    def get_optimizer_params(self, fsdp_model):
        """Return the parameters to optimize."""

    @abstractmethod
    def compute_speculator_loss(
        self,
        fsdp_model,
        input_ids,
        attention_mask=None,
        position_ids=None,
        loss_mask=None,
        hidden_states=None,
        spec_logits=None,
    ):
        """Compute speculator loss."""

    def _maybe_shard_speculator(self, fsdp_model, speculator_module, fsdp_kwargs: Optional[dict]):
        if speculator_module is None:
            return speculator_module
        if fully_shard is None:
            return speculator_module
        if not fsdp_kwargs:
            raise ValueError("fsdp_kwargs must be provided when sharding speculator module")
        with maybe_patch_fsdp_module(speculator_module):
            speculator_module = fully_shard(speculator_module, **fsdp_kwargs)
        fsdp_model.speculator = speculator_module
        self.speculator = speculator_module
        return speculator_module

    def shard_speculator(self, fsdp_model, fsdp_kwargs: Optional[dict]):
        speculator_module = self._get_speculator_module(fsdp_model)
        return self._maybe_shard_speculator(fsdp_model, speculator_module, fsdp_kwargs)

    def _get_speculator_module(self, fsdp_model):
        if fsdp_model is not None and hasattr(fsdp_model, "speculator"):
            return fsdp_model.speculator
        if hasattr(self, "speculator"):
            return self.speculator
        return None

    def _get_speculator_config_obj(self, fsdp_model, speculator_module):
        if speculator_module is None:
            return None
        return getattr(speculator_module, "config", None)

    def _maybe_pad_nested(self, tensor, padding):
        if isinstance(tensor, torch.Tensor) and tensor.is_nested:
            return torch.nested.to_padded_tensor(tensor, padding=padding)
        return tensor

    def _slice_speculator_inputs(self, input_ids, hidden_states, n_predict):
        # Alignment rule: hidden_states[t] should predict tokens starting at input_ids[t+1],
        # and we need n_predict extra tokens to the right, so drop the last n_predict+1 states.
        hidden = hidden_states[:, : -(n_predict + 1), :]
        seq_ids = input_ids[:, 1:]
        return hidden, seq_ids

    def save_checkpoint(self, fsdp_model, local_global_step_folder: str):
        if not getattr(self, "has_speculator", False):
            return
        speculator_module = self._get_speculator_module(fsdp_model)
        if speculator_module is None:
            return
        speculator_dir = os.path.join(local_global_step_folder, "speculator")
        config_obj = self._get_speculator_config_obj(fsdp_model, speculator_module)
        save_speculator_checkpoint(
            fsdp_model,
            speculator_module,
            speculator_dir,
            config_obj=config_obj,
        )

    def load_checkpoint(self, fsdp_model, checkpoint_path: str, logger):
        if not getattr(self, "has_speculator", False):
            return
        speculator_module = self._get_speculator_module(fsdp_model)
        if speculator_module is None:
            return
        load_speculator_checkpoint(
            fsdp_model,
            speculator_module,
            checkpoint_path,
            logger=logger,
            device_mesh=getattr(self, "device_mesh", None),
        )


def _load_custom_adapter(speculator_adapter_config: Any):
    if isinstance(speculator_adapter_config, str):
        return load_class_from_fqn(speculator_adapter_config, description="speculator adapter")

    if hasattr(speculator_adapter_config, "get"):
        fqn = speculator_adapter_config.get("fqn", None)
        if fqn:
            return load_class_from_fqn(fqn, description="speculator adapter")
        path = speculator_adapter_config.get("path", None)
        name = speculator_adapter_config.get("name", None)
        if path and name:
            return load_extern_object(path, name)

    return None


def build_speculator_adapter(
    config,
    model_config,
    device_name,
    device_mesh,
    torch_dtype,
):
    speculator_adapter_config = None
    if config is not None and hasattr(config, "model"):
        speculator_adapter_config = getattr(config.model, "speculator_adapter", None)
    if speculator_adapter_config is None and model_config is not None:
        speculator_adapter_config = getattr(model_config, "speculator_adapter", None)

    adapter_cls = _load_custom_adapter(speculator_adapter_config)
    if adapter_cls is None:
        raise ValueError(
            "speculator_adapter is not configured. Please set config.model.speculator_adapter "
            "to a valid adapter class or module path."
        )

    return adapter_cls(
        config=config,
        model_config=model_config,
        device_name=device_name,
        device_mesh=device_mesh,
        torch_dtype=torch_dtype,
    )


class SpeculatorManager:
    def __init__(
        self,
        config,
        model_config,
        device_name,
        device_mesh,
        torch_dtype,
    ):
        self.config = config
        self.model_config = model_config
        self.device_name = device_name
        self.device_mesh = device_mesh
        self.torch_dtype = torch_dtype
        self.adapter: Optional[SpeculatorAdapter] = None
        self.speculator = None
        self.has_speculator = False
        self._build_adapter()

    def _build_adapter(self) -> None:
        speculator_config = None
        if self.config is not None and hasattr(self.config, "model"):
            speculator_config = getattr(self.config.model, "speculator", None)
            speculator_adapter_config = getattr(self.config.model, "speculator_adapter", None)
        else:
            speculator_adapter_config = getattr(self.model_config, "speculator_adapter", None)

        if speculator_config is None and speculator_adapter_config is None:
            return

        self.adapter = build_speculator_adapter(
            self.config,
            self.model_config,
            self.device_name,
            self.device_mesh,
            self.torch_dtype,
        )
        self.has_speculator = self.adapter.has_speculator

    def attach(self, fsdp_strategy, model=None, fsdp_model=None, fsdp_kwargs=None):
        if self.adapter is None:
            return None
        if fsdp_strategy == "fsdp":
            self.speculator = self.adapter.build_and_attach(model, attach_to_model=True)
            return self.speculator
        if fsdp_strategy == "fsdp2":
            self.speculator = self.adapter.build_and_attach(fsdp_model, attach_to_model=True)
            self.speculator = self.adapter.shard_speculator(fsdp_model, fsdp_kwargs)
            return self.speculator
        raise NotImplementedError(f"not implement {fsdp_strategy}")

    def build_speculator(self, model=None, fsdp_model=None):
        if self.adapter is None:
            return None
        target_model = fsdp_model if fsdp_model is not None else model
        if target_model is None:
            return None
        self.speculator = self.adapter.build_and_attach(target_model, attach_to_model=True)
        return self.speculator

    def shard_speculator(self, fsdp_model, fsdp_kwargs=None):
        if self.adapter is None:
            return None
        self.speculator = self.adapter.shard_speculator(fsdp_model, fsdp_kwargs)
        return self.speculator

    def get_optimizer_params(self, fsdp_model):
        if self.adapter is None:
            return fsdp_model.parameters()
        return self.adapter.get_optimizer_params(fsdp_model)

    def compute_speculator_loss(
        self,
        fsdp_model: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        spec_logits: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.adapter is None:
            return None
        return self.adapter.compute_speculator_loss(
            fsdp_model,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            hidden_states=hidden_states,
            spec_logits=spec_logits,
        )

    def save_checkpoint(self, fsdp_model, local_global_step_folder: str) -> None:
        if self.adapter is None:
            return
        self.adapter.save_checkpoint(fsdp_model, local_global_step_folder)

    def load_checkpoint(self, fsdp_model, checkpoint_path: str, logger) -> None:
        if self.adapter is None:
            return
        self.adapter.load_checkpoint(fsdp_model, checkpoint_path, logger)
