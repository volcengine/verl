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
"""Utilities for selecting and loading models."""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from vllm.config import CacheConfig, DeviceConfig, LoadConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.loader import _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from .config import LoadConfig, LoadFormat, ModelConfig
from .dtensor_weight_loaders import load_dtensor_weights, update_dtensor_weight_loader
from .hf_weight_loader import update_hf_weight_loader
from .megatron_weight_loaders import load_megatron_weights, update_megatron_weight_loader


def get_model(
    actor_model: Union[PreTrainedModel, Dict],
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    lora_config: Optional[LoRAConfig],
    cache_config: CacheConfig = None,
) -> nn.Module:
    loader = get_model_loader(load_config)
    if load_config.load_format.startswith("dummy"):
        return loader.load_model(
            model_config=model_config,
            device_config=device_config,
            lora_config=lora_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
        )
    else:
        return loader.load_model(
            actor_model=actor_model,
            model_config=model_config,
            device_config=device_config,
            lora_config=lora_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
        )


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.AUTO:
        update_megatron_weight_loader()
        return MegatronLoader(load_config)

    # NOTE(sgm): change the weight_loader function in runtime
    if load_config.load_format == LoadFormat.MEGATRON:
        update_megatron_weight_loader()
        return MegatronLoader(load_config)

    if load_config.load_format == LoadFormat.HF:
        update_hf_weight_loader()
        return HFLoader(load_config)

    if load_config.load_format == LoadFormat.DTENSOR:
        update_dtensor_weight_loader()
        return DTensorLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_HF:
        update_hf_weight_loader()
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_MEGATRON:
        update_megatron_weight_loader()
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.DUMMY_DTENSOR:
        update_dtensor_weight_loader()
        return DummyModelLoader(load_config)

    raise ValueError("load format not supported in verl: {}, only support {} and {}".format(
        load_config.load_format, LoadFormat.MEGATRON, LoadFormat.HF))


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            # initialize_dummy_weights(model)
        return model.eval()


class MegatronLoader(BaseModelLoader):
    """Model loader that can load the model weights from partitioned megatron model."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        # NOTE(shengguangming) Load the weights from the actor model
        pass
        # if isinstance(actor_model, nn.Module):
        #     load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
        # else:
        #     load_weights(actor_weights=actor_model, vllm_model=model)
        # return actor_model

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)

            # TODO(sgm): This is a hack, we need to register the load_weight() func for each model in vllm
            if isinstance(actor_model, nn.Module):
                load_megatron_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)),
                                      vllm_model=model)
            else:
                load_megatron_weights(actor_weights=actor_model, vllm_model=model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        return model.eval()


class HFLoader(BaseModelLoader):
    """Model loader that can load the model weights from model's full params."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def _get_weights_iterator(self, actor_model: Union[PreTrainedModel, Dict]):
        if isinstance(actor_model, Dict):
            return actor_model.items()
        elif isinstance(actor_model, nn.Module):
            return dict(actor_model.named_parameters()).items()
        else:
            raise ValueError(f"actor model should be Dict or nn.Module, but get {type(actor_model)}")

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            # with torch.device(device_config.device):
            # NOTE(sgm): init the model in cpu
            model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)
            model.load_weights(self._get_weights_iterator(actor_model))
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        return model.eval()


class DTensorLoader(BaseModelLoader):
    """Model loader that can load the model weights from partitioned megatron model."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        # NOTE(shengguangming) Load the weights from the actor model
        pass
        # if isinstance(actor_model, nn.Module):
        #     load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
        # else:
        #     load_weights(actor_weights=actor_model, vllm_model=model)
        # return actor_model

    def load_model(
        self,
        actor_model: Union[PreTrainedModel, Dict],
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)

            # TODO(sgm): This is a hack, we need to register the load_weight() func for each model in vllm
            if isinstance(actor_model, nn.Module):
                load_dtensor_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)),
                                     vllm_model=model)
            else:
                load_dtensor_weights(actor_weights=actor_model, vllm_model=model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        return model.eval()


# FIXME(sgm): hack the _get_logits function in vllm v0.4.2
# as they use ray, the _get_logits result will only need to return to the driver node,
# therefore gather is enough. However, we use SPMD instead of a central scheduler,
# all_gather is required (aligned with v0.2.6)
def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_all_gather(logits)
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, :self.org_vocab_size]
    return logits


from vllm.model_executor.layers.logits_processor import LogitsProcessor


def logitsprocessor_init(
    self,
    vocab_size: int,
    org_vocab_size: Optional[int] = None,
    scale: float = 1.0,
    logits_as_input: bool = False,
    soft_cap: Optional[float] = None,
) -> None:
    """
    Args:
        scale: A scaling factor to apply to the logits.
    """
    super(LogitsProcessor, self).__init__()
    self.scale = scale
    self.vocab_size = vocab_size
    # Whether the input is logits (default is hidden states).
    self.logits_as_input = logits_as_input
    # original vocabulary size (without LoRA).
    self.org_vocab_size = org_vocab_size or vocab_size
    # Soft cap the logits. Used in Gemma 2.
    self.soft_cap = soft_cap
    # Whether to use gather or all-gather to gather the logits.
    self.use_gather = False


LogitsProcessor.__init__ = logitsprocessor_init  # use all_gather
