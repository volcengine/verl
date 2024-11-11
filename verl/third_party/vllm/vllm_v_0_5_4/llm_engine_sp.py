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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py

import torch
from typing import Dict, Optional, Union, Type

import vllm.envs as envs
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, MultiModalConfig,
                         ObservabilityConfig, ParallelConfig, PromptAdapterConfig, SchedulerConfig, SpeculativeConfig)
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import INPUT_REGISTRY, LLMInputs, PromptInputs
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.engine.metrics import (LoggingStatLogger, PrometheusStatLogger, StatLoggerBase, Stats)
from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context, init_tracer)
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled, usage_message)
from vllm.utils import Counter
from vllm.engine.llm_engine import _load_generation_config_dict
from vllm.engine.llm_engine import LLMEngine
from vllm.version import __version__ as VLLM_VERSION

import torch.nn as nn
from .arg_utils import EngineArgs
from .tokenizer import TokenizerGroup
from .config import ModelConfig, LoadConfig

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class LLMEngine(LLMEngine):
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: the actor model initialize outside vllm (add for verl)
        tokenizer: the initialized tokenizer (add for verl)
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        # NOTE(sgm): first two arguments are added for verl
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        tokenizer: nn.Module,
        # NOTE(sgm): vllm original arguments
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> None:
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, revision=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, use_v2_block_manager=%s, "
            "enable_prefix_caching=%s)",
            VLLM_VERSION,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.revision,
            model_config.rope_scaling,
            model_config.rope_theta,
            model_config.tokenizer_revision,
            model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            observability_config,
            model_config.seed,
            model_config.served_model_name,
            scheduler_config.use_v2_block_manager,
            cache_config.enable_prefix_caching,
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.multimodal_config = multimodal_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config or ObservabilityConfig()
        self.log_stats = log_stats

        # self.model = model # should not store the model, it should be deleted
        # TODO(shengguangming): maybe we can choose init here or from arguments
        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer(tokenizer)
            self.detokenizer = Detokenizer(self.tokenizer)
        else:
            self.tokenizer = None
            self.detokenizer = None

        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(model_config)

        self.input_processor = INPUT_REGISTRY.create_input_processor(self.model_config)

        self.model_executor = executor_class(
            model=model, # add for spmd_gpu_executor
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            multimodal_config=multimodal_config,
            speculative_config=speculative_config,
            load_config=load_config,
            prompt_adapter_config=prompt_adapter_config,
        )

        # Profile the memory usage and initialize the cache.
        if not self.model_config.embedding_mode:
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype": str(model_config.dtype),
                    "tensor_parallel_size": parallel_config.tensor_parallel_size,
                    "block_size": cache_config.block_size,
                    "gpu_memory_utilization": cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization": model_config.quantization,
                    "kv_cache_dtype": str(cache_config.cache_dtype),

                    # Feature flags
                    "enable_lora": bool(lora_config),
                    "enable_prompt_adapter": bool(prompt_adapter_config),
                    "enable_prefix_caching": cache_config.enable_prefix_caching,
                    "enforce_eager": model_config.enforce_eager,
                    "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = [
            Scheduler(scheduler_config, cache_config, lora_config, parallel_config.pipeline_parallel_size)
            for _ in range(parallel_config.pipeline_parallel_size)
        ]

        # Metric Logging.
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                self.stat_loggers = {
                    "logging":
                        LoggingStatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC),
                    "prometheus":
                        PrometheusStatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                                             labels=dict(model_name=model_config.served_model_name),
                                             max_model_len=self.model_config.max_model_len),
                }
                self.stat_loggers["prometheus"].info("cache_config", self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer("vllm.llm_engine", self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (SequenceGroupOutputProcessor.create_output_processor(
            self.scheduler_config,
            self.detokenizer,
            self.scheduler,
            self.seq_counter,
            self.get_tokenizer_for_seq,
            stop_checker=StopChecker(
                self.scheduler_config.max_model_len,
                self.get_tokenizer_for_seq,
            ),
        ))

    # TODO(sgm): add for verl but we may not tokenizer in Rollout
    def _init_tokenizer(self, tokenizer, **tokenizer_init_kwargs):
        init_kwargs = dict(enable_lora=bool(self.lora_config),
                           max_num_seqs=self.scheduler_config.max_num_seqs,
                           max_input_length=None)
        init_kwargs.update(tokenizer_init_kwargs)
        return TokenizerGroup(tokenizer, **init_kwargs)

    def init_cache_engine(self):
        # TODO: check whether we should rebuild the CUDAGraph every iter when offload/load KVCache
        # Re-capture CUDAGraph would be time-consuming
        self.model_executor.init_cache_engine()

    def free_cache_engine(self):
        self.model_executor.free_cache_engine()

    # NOTE(sgm): currently, we only support GPU executor
    # The GPUExecutor remove the Ray dependency
    @classmethod
    def _get_executor_cls(cls, engine_config: EngineConfig) -> Type[ExecutorBase]:
        assert engine_config.device_config.device_type == "cuda", \
            "Currently, the vllm in verl only support running on GPU"

        if engine_config.parallel_config.world_size == 1:
            engine_config.load_config.load_format = "dummy_hf"

        from .spmd_gpu_executor import SPMDGPUExecutor
        executor_class = SPMDGPUExecutor
        return executor_class

    @classmethod
    def from_engine_args(
        cls,
        model,
        tokenizer,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        executor_class = cls._get_executor_cls(engine_config)
        # Initialize the cluster and specify the executor class.
        assert engine_config.device_config.device_type == "cuda", \
            "Currently, the vllm in verl only support running on GPU"

        from .spmd_gpu_executor import SPMDGPUExecutor
        executor_class = SPMDGPUExecutor

        # Create the LLM engine.
        engine = cls(
            model,
            tokenizer,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.model_executor.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.model_executor.offload_model_weights()
