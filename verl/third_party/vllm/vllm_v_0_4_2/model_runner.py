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

from vllm.attention import (AttentionMetadata, get_attn_backend)
from vllm.config import (DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.sequence import (MultiModalData, SamplerOutput, SequenceData, SequenceGroupMetadata)
from vllm.utils import (CudaMemoryProfiler, is_hip, is_pin_memory_available)
from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner

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
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.load_config = load_config

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window() if model_config is not None else None)
        self.device_config = (device_config if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        # NOTE(sgm): add for verl
        self.model = model  # this will be replaced by get_model()

        # Set after load_model.
        self.lora_manager: LRUCacheWorkerLoRAManager = None

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[int, int]] = None  # Set during graph capture.

        self.max_seq_len_to_capture = (self.model_config.max_seq_len_to_capture if self.model_config is not None else 0)

        self.pin_memory = is_pin_memory_available()
        self.kv_cache_dtype = kv_cache_dtype
        self.vision_language_config = vision_language_config

        self.attn_backend = get_attn_backend(self.model_config.dtype if model_config is not None else None)

        # Lazy initialization
        self.block_size: int  # Set after initial profiling.
        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables: torch.Tensor  # Set after initial profiling.

        # Set if the backend is flashinfer.
        self.flashinfer_workspace_buffer: torch.Tensor

    # NOTE(sgm): initialize model using the actor model
    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(actor_model=self.model,
                                   model_config=self.model_config,
                                   device_config=self.device_config,
                                   lora_config=self.lora_config,
                                   load_config=self.load_config,
                                   parallel_config=self.parallel_config,
                                   scheduler_config=self.scheduler_config,
                                   vision_language_config=self.vision_language_config)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert hasattr(self.model, "supported_lora_modules") and self.model.supported_lora_modules, (
                "Model does not support LoRA")
            assert hasattr(self.model, "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(self.scheduler_config.max_num_seqs,
                                                          self.scheduler_config.max_num_batched_tokens, self.vocab_size,
                                                          self.lora_config, self.device, self.model.embedding_modules,
                                                          self.model.embedding_padding_modules)
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently scaled KV cache is only enabled on ROCm
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.", self.model.__class__)
            else:
                logger.warning("Using FP8 KV cache but no scaling factors "
                               "provided. Defaulting to scaling factors of 1.0. "
                               "This may lead to less accurate results!")
        elif self.model_config.quantization_param_path is not None:
            logger.warning("KV cache scaling factors provided, "
                           "but the KV cache data type is not FP8. "
                           "KV cache scaling factors will not be used.")

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata, Set[LoRARequest], LoRAMapping,
               torch.Tensor]:
        # NOTE(sgm): all workers prepare the input in the same way
        prefill_reqs = []
        decode_reqs = []
        for seq_group_meta in seq_group_metadata_list:
            if seq_group_meta.is_prompt:
                prefill_reqs.append(seq_group_meta)
            else:
                decode_reqs.append(seq_group_meta)

        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            prefill_attn_metadata,
            seq_lens,
            query_lens,
            lora_index_mapping,
            lora_prompt_mapping,
            lora_requests,
            multi_modal_input,
            slot_mapping,
        ) = self._prepare_prompt(prefill_reqs)
        (
            decode_input_tokens,
            decode_input_positions,
            decode_attn_metadata,
            decode_lora_index_mapping,
            decode_lora_prompt_mapping,
            decode_lora_requests,
            decode_slot_mapping,
        ) = self._prepare_decode(decode_reqs)
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list, seq_lens, query_lens, self.device,
                                                     self.pin_memory)

        if not self.scheduler_config.chunked_prefill_enabled:
            assert (len(prefill_reqs) and len(decode_reqs)) == 0

        num_prefills = len(seq_lens)
        num_prefill_tokens = len(input_tokens)
        num_decode_tokens = len(decode_input_tokens)

        # Coalesce tensors. Note that attn_metadata is currently not
        # coalesced for simplicity.
        input_tokens.extend(decode_input_tokens)
        input_positions.extend(decode_input_positions)
        slot_mapping.extend(decode_slot_mapping)
        lora_index_mapping.extend(decode_lora_index_mapping)
        lora_prompt_mapping.extend(decode_lora_prompt_mapping)
        lora_requests.update(decode_lora_requests)

        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        input_positions = torch.tensor(input_positions, dtype=torch.long, device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)

        if self.lora_config:
            lora_mapping = LoRAMapping(
                lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        # Broadcast the metadata.
        # If batch contains both prefill and decode, it sends 2 broadcasts.
        # If it only contains 1 type, it triggers a single broadcast.
        if (prefill_attn_metadata is not None and decode_attn_metadata is not None):
            batch_type = BatchType.MIXED
        elif prefill_attn_metadata is not None:
            batch_type = BatchType.PREFILL
        else:
            batch_type = BatchType.DECODE

        attn_metadata = AttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_attn_metadata,
            decode_metadata=decode_attn_metadata,
            kv_cache_dtype=self.kv_cache_dtype,
        )

        return (input_tokens, input_positions, attn_metadata, sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata, lora_requests, lora_mapping,
         multi_modal_input) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        # if not self.is_driver_worker:
        #     return None

        # TODO(sgm): perform sampling on rank 0
        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return output
