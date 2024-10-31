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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py

import os
import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

import torch.nn as nn

from transformers import PretrainedConfig
from .config import ModelConfig, LoadConfig

from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, MultiModalConfig,
                         ObservabilityConfig, ParallelConfig, PromptAdapterConfig, SchedulerConfig, SpeculativeConfig,
                         TokenizerPoolConfig)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import str_to_int_tuple

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (BaseTokenizerGroup)

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model_hf_config: PretrainedConfig = None  # for verl
    served_model_name = None  # TODO(sgm): check this
    # tokenizer: Optional[str] = None # TODO(sgm): check this
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[str, Type[ExecutorBase]]] = None
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = False
    swap_space: int = 4  # GiB
    cpu_offload_gb: int = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: Optional[int] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = 1
    max_prompt_adapter_token: int = 0
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: str = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    guided_decoding_backend: str = 'outlines'
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    qlora_adapter_name_or_path: Optional[str] = None
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    otlp_traces_endpoint: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        # TODO(shengguangming): delete the unused args
        parser.add_argument('--model',
                            type=str,
                            default='facebook/opt-125m',
                            help='name or path of the huggingface model to use')
        parser.add_argument('--tokenizer',
                            type=str,
                            default=EngineArgs.tokenizer,
                            help='name or path of the huggingface tokenizer to use')
        parser.add_argument('--revision',
                            type=str,
                            default=None,
                            help='the specific model version to use. It can be a branch '
                            'name, a tag name, or a commit id. If unspecified, will use '
                            'the default version.')
        parser.add_argument('--tokenizer-revision',
                            type=str,
                            default=None,
                            help='the specific tokenizer version to use. It can be a branch '
                            'name, a tag name, or a commit id. If unspecified, will use '
                            'the default version.')
        parser.add_argument('--tokenizer-mode',
                            type=str,
                            default=EngineArgs.tokenizer_mode,
                            choices=['auto', 'slow'],
                            help='tokenizer mode. "auto" will use the fast '
                            'tokenizer if available, and "slow" will '
                            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code', action='store_true', help='trust remote code from huggingface')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface')
        parser.add_argument('--load-format',
                            type=str,
                            default=EngineArgs.load_format,
                            choices=['auto', 'pt', 'safetensors', 'npcache', 'dummy'],
                            help='The format of the model weights to load. '
                            '"auto" will try to load the weights in the safetensors format '
                            'and fall back to the pytorch bin format if safetensors format '
                            'is not available. '
                            '"pt" will load the weights in the pytorch bin format. '
                            '"safetensors" will load the weights in the safetensors format. '
                            '"npcache" will load the weights in pytorch format and store '
                            'a numpy cache to speed up the loading. '
                            '"dummy" will initialize the weights with random values, '
                            'which is mainly for profiling.')
        parser.add_argument('--dtype',
                            type=str,
                            default=EngineArgs.dtype,
                            choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                            help='data type for model weights and activations. '
                            'The "auto" option will use FP16 precision '
                            'for FP32 and FP16 models, and BF16 precision '
                            'for BF16 models.')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=None,
                            help='model context length. If unspecified, '
                            'will be automatically derived from the model.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed', type=int, default=EngineArgs.seed, help='random seed')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the model executor')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats', action='store_true', help='disable logging statistics')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=str,
                            choices=['awq', None],
                            default=None,
                            help='Method used to quantize the weights')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_config(
        self,
    ) -> EngineConfig:
        # bitsandbytes quantization needs a specific model loader
        # so we make sure the quant method and the load format are consistent
        if (self.quantization == "bitsandbytes" or
           self.qlora_adapter_name_or_path is not None) and \
           self.load_format != "bitsandbytes":
            raise ValueError("BitsAndBytes quantization and QLoRA adapter only support "
                             f"'bitsandbytes' load format, but got {self.load_format}")

        if (self.load_format == "bitsandbytes" or
            self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            raise ValueError("BitsAndBytes load format and QLoRA adapter only support "
                             f"'bitsandbytes' quantization, but got {self.quantization}")

        assert self.cpu_offload_gb >= 0, ("CPU offload space must be non-negative"
                                          f", but got {self.cpu_offload_gb}")

        multimodal_config = MultiModalConfig()
        device_config = DeviceConfig(self.device)
        # NOTE(sgm): we only modify ModelConfig, other configs are import from vllm
        model_config = ModelConfig(hf_config=self.model_hf_config,
                                   tokenizer_mode=self.tokenizer_mode,
                                   trust_remote_code=self.trust_remote_code,
                                   dtype=self.dtype,
                                   seed=self.seed,
                                   revision=self.revision,
                                   code_revision=self.code_revision,
                                   rope_scaling=self.rope_scaling,
                                   rope_theta=self.rope_theta,
                                   tokenizer_revision=self.tokenizer_revision,
                                   max_model_len=self.max_model_len,
                                   quantization=self.quantization,
                                   quantization_param_path=self.quantization_param_path,
                                   enforce_eager=self.enforce_eager,
                                   max_context_len_to_capture=self.max_context_len_to_capture,
                                   max_seq_len_to_capture=self.max_seq_len_to_capture,
                                   max_logprobs=self.max_logprobs,
                                   disable_sliding_window=self.disable_sliding_window,
                                   skip_tokenizer_init=self.skip_tokenizer_init,
                                   served_model_name=self.served_model_name,
                                   multimodal_config=multimodal_config)
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
            cpu_offload_gb=self.cpu_offload_gb,
        )
        parallel_config = ParallelConfig(pipeline_parallel_size=self.pipeline_parallel_size,
                                         tensor_parallel_size=self.tensor_parallel_size,
                                         worker_use_ray=self.worker_use_ray,
                                         max_parallel_loading_workers=self.max_parallel_loading_workers,
                                         disable_custom_all_reduce=self.disable_custom_all_reduce,
                                         tokenizer_pool_config=TokenizerPoolConfig.create_config(
                                             self.tokenizer_pool_size,
                                             self.tokenizer_pool_type,
                                             self.tokenizer_pool_extra_config,
                                         ),
                                         ray_workers_use_nsight=self.ray_workers_use_nsight,
                                         distributed_executor_backend=self.distributed_executor_backend)

        # NOTE[VERL]: Use the world_size set by TORCHRUN
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
        parallel_config.world_size = world_size

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # If not explicitly set, enable chunked prefill by default for
            # long context (> 32K) models. This is to avoid OOM errors in the
            # initial memory profiling phase.
            if use_long_context:
                is_gpu = device_config.device_type == "cuda"
                use_sliding_window = (model_config.get_sliding_window() is not None)
                use_spec_decode = self.speculative_model is not None
                has_seqlen_agnostic_layers = (model_config.contains_seqlen_agnostic_layers(parallel_config))
                if (is_gpu and not use_sliding_window and not use_spec_decode and not self.enable_lora and
                        not self.enable_prompt_adapter and not self.enable_prefix_caching and
                        not has_seqlen_agnostic_layers):
                    self.enable_chunked_prefill = True
                    logger.warning("Chunked prefill is enabled by default for models with "
                                   "max_model_len > 32K. Currently, chunked prefill might "
                                   "not work with some features or models. If you "
                                   "encounter any issues, please disable chunked prefill "
                                   "by setting --enable-chunked-prefill=False.")
            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause OOM "
                "errors during the initial memory profiling phase, or result "
                "in low performance due to small KV cache space. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)

        # TODO: spec config
        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            target_dtype=self.dtype,
            speculative_model=self.speculative_model,
            speculative_draft_tensor_parallel_size = \
                self.speculative_draft_tensor_parallel_size,
            num_speculative_tokens=self.num_speculative_tokens,
            speculative_disable_by_batch_size=self.
            speculative_disable_by_batch_size,
            speculative_max_model_len=self.speculative_max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            use_v2_block_manager=self.use_v2_block_manager,
            disable_log_stats=self.disable_log_stats,
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
            draft_token_acceptance_method=\
                self.spec_decoding_acceptance_method,
            typical_acceptance_sampler_posterior_threshold=self.
            typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=self.
            typical_acceptance_sampler_posterior_alpha,
            disable_logprobs=self.disable_logprobs_during_spec_decoding,
        )

        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            use_v2_block_manager=self.use_v2_block_manager,
            num_lookahead_slots=(self.num_lookahead_slots
                                 if speculative_config is None else speculative_config.num_lookahead_slots),
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            embedding_mode=model_config.embedding_mode,
            preemption_mode=self.preemption_mode,
        )
        lora_config = LoRAConfig(max_lora_rank=self.max_lora_rank,
                                 max_loras=self.max_loras,
                                 fully_sharded_loras=self.fully_sharded_loras,
                                 lora_extra_vocab_size=self.lora_extra_vocab_size,
                                 long_lora_scaling_factors=self.long_lora_scaling_factors,
                                 lora_dtype=self.lora_dtype,
                                 max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras and self.max_cpu_loras > 0 else
                                 None) if self.enable_lora else None

        if self.qlora_adapter_name_or_path is not None and \
            self.qlora_adapter_name_or_path != "":
            if self.model_loader_extra_config is None:
                self.model_loader_extra_config = {}
            self.model_loader_extra_config["qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

        prompt_adapter_config = PromptAdapterConfig(
            max_prompt_adapters=self.max_prompt_adapters,
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        decoding_config = DecodingConfig(guided_decoding_backend=self.guided_decoding_backend)

        observability_config = ObservabilityConfig(otlp_traces_endpoint=self.otlp_traces_endpoint)

        if (model_config.get_sliding_window() is not None and scheduler_config.chunked_prefill_enabled and
                not scheduler_config.use_v2_block_manager):
            raise ValueError("Chunked prefill is not supported with sliding window. "
                             "Set --disable-sliding-window to disable sliding window.")

        return EngineConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            multimodal_config=multimodal_config,
            speculative_config=speculative_config,
            load_config=load_config,
            decoding_config=decoding_config,
            observability_config=observability_config,
            prompt_adapter_config=prompt_adapter_config,
        )
