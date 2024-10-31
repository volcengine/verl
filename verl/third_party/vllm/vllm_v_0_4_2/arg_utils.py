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
from dataclasses import dataclass
from typing import List, Optional, Union

import torch.nn as nn

from transformers import PretrainedConfig
from .config import ModelConfig, LoadConfig

from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, TokenizerPoolConfig, VisionLanguageConfig)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import str_to_int_tuple


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model_hf_config: PretrainedConfig = None
    skip_tokenizer_init: bool = False
    served_model_name: Optional[Union[str, List[str]]] = None  # TODO
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    use_v2_block_manager: bool = False
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 5  # OpenAI default value
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: Optional[int] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    tokenizer_pool_type: str = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    lora_dtype = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None

    # Related to Vision-language models such as llava
    image_input_type: Optional[str] = None
    image_token_id: Optional[int] = None
    image_input_shape: Optional[str] = None
    image_feature_size: Optional[int] = None
    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: bool = False

    guided_decoding_backend: str = 'outlines'
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None
    speculative_max_model_len: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None

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
        device_config = DeviceConfig(self.device)
        # NOTE(sgm): we only modify ModelConfig, other configs are import from vllm
        model_config = ModelConfig(self.model_hf_config, self.dtype, self.seed, self.revision, self.code_revision,
                                   self.tokenizer_revision, self.max_model_len, self.quantization,
                                   self.quantization_param_path, self.enforce_eager, self.max_context_len_to_capture,
                                   self.max_seq_len_to_capture, self.max_logprobs, self.skip_tokenizer_init,
                                   self.served_model_name)
        cache_config = CacheConfig(self.block_size, self.gpu_memory_utilization,
                                   self.swap_space, self.kv_cache_dtype, self.num_gpu_blocks_override,
                                   model_config.get_sliding_window(), self.enable_prefix_caching)
        parallel_config = ParallelConfig(
            self.pipeline_parallel_size, self.tensor_parallel_size, self.worker_use_ray,
            self.max_parallel_loading_workers, self.disable_custom_all_reduce,
            TokenizerPoolConfig.create_config(
                self.tokenizer_pool_size,
                self.tokenizer_pool_type,
                self.tokenizer_pool_extra_config,
            ), self.ray_workers_use_nsight)

        # Use the world_size set by TORCHRUN
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
        parallel_config.world_size = world_size

        # TODO: spec config
        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            target_dtype=self.dtype,
            speculative_model=self.speculative_model,
            num_speculative_tokens=self.num_speculative_tokens,
            speculative_max_model_len=self.speculative_max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            use_v2_block_manager=self.use_v2_block_manager,
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
        )

        scheduler_config = SchedulerConfig(
            self.max_num_batched_tokens,
            self.max_num_seqs,
            model_config.max_model_len,
            self.use_v2_block_manager,
            num_lookahead_slots=(self.num_lookahead_slots
                                 if speculative_config is None else speculative_config.num_lookahead_slots),
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
        )

        lora_config = LoRAConfig(max_lora_rank=self.max_lora_rank,
                                 max_loras=self.max_loras,
                                 fully_sharded_loras=self.fully_sharded_loras,
                                 lora_extra_vocab_size=self.lora_extra_vocab_size,
                                 lora_dtype=self.lora_dtype,
                                 max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras and self.max_cpu_loras > 0 else
                                 None) if self.enable_lora else None

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
        )

        if self.image_input_type:
            if (not self.image_token_id or not self.image_input_shape or not self.image_feature_size):
                raise ValueError('Specify `image_token_id`, `image_input_shape` and '
                                 '`image_feature_size` together with `image_input_type`.')
            vision_language_config = VisionLanguageConfig(
                image_input_type=VisionLanguageConfig.get_image_input_enum_type(self.image_input_type),
                image_token_id=self.image_token_id,
                image_input_shape=str_to_int_tuple(self.image_input_shape),
                image_feature_size=self.image_feature_size,
            )
        else:
            vision_language_config = None

        decoding_config = DecodingConfig(guided_decoding_backend=self.guided_decoding_backend)

        return EngineConfig(model_config=model_config,
                            cache_config=cache_config,
                            parallel_config=parallel_config,
                            scheduler_config=scheduler_config,
                            device_config=device_config,
                            lora_config=lora_config,
                            vision_language_config=vision_language_config,
                            speculative_config=speculative_config,
                            load_config=load_config,
                            decoding_config=decoding_config)
