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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py

import torch
import torch.nn as nn
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

import vllm.envs as envs
from vllm.attention import (AttentionMetadata, get_attn_backend)
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.models.interfaces import (supports_lora, supports_vision)
from vllm.utils import (CudaMemoryProfiler, is_hip, is_pin_memory_available)
from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner
from vllm.prompt_adapter.worker_manager import (LRUCacheWorkerPromptAdapterManager)

from .model_loader import get_model
from .config import ModelConfig, LoadConfig

logger = init_logger(__name__)


# How batches are constructed.
class BatchType(IntEnum):
    # Every batch is prefill.
    PREFILL = 0
    # Every batch is decode.
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    MIXED = 2


class ModelRunner(ModelRunner):

    def __init__(
        self,
        model: Union[nn.Module, Dict], # [verl] model itself or its parameter dict
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        return_hidden_states: bool = False,
    ):

        super().__init__(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config,
            lora_config,
            kv_cache_dtype,
            is_driver_worker=True,  # a hack
            prompt_adapter_config=prompt_adapter_config,
            multimodal_config=multimodal_config,
            return_hidden_states=return_hidden_states)

        # NOTE(sgm): add for verl
        self.model = model  # this will be replaced by get_model()

    # NOTE(sgm): initialize model using the actor model
    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with CudaMemoryProfiler() as m:
            self.model = get_model(actor_model=self.model,
                                   model_config=self.model_config,
                                   device_config=self.device_config,
                                   lora_config=self.lora_config,
                                   load_config=self.load_config,
                                   parallel_config=self.parallel_config,
                                   scheduler_config=self.scheduler_config,
                                   multimodal_config=self.multimodal_config,
                                   cache_config=self.cache_config)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert supports_lora(self.model), "Model does not support LoRA"
            assert not supports_vision(self.model), "To be tested: vision language model with LoRA settings."

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=self.model.config.max_position_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs, self.scheduler_config.max_num_batched_tokens, self.device,
                self.prompt_adapter_config)
            self.model = (self.prompt_adapter_manager.create_prompt_adapter_manager(self.model))

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s", self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.", self.model.__class__)
            else:
                logger.warning("Using FP8 KV cache but no scaling factors "
                               "provided. Defaulting to scaling factors of 1.0. "
                               "This may lead to less accurate results!")

        if envs.VLLM_TEST_DYNAMO_GRAPH_CAPTURE:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")
