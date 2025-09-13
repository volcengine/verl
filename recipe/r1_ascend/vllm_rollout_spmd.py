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
import gc
import logging
import os

import torch
import torch.distributed
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout import vLLMRollout as vLLMRolloutBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMRollout(vLLMRolloutBase):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        self.config = config
        # NPU-ADAPTATION: import vLLM-Ascend patch
        from vllm_ascend.patch import platform
        from vllm_ascend.patch import worker
        from recipe.r1_ascend import engine_core
        # NPU-ADAPTATION END

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)
        
        # NPU-ADAPTATION: VLLM_DP_SIZE is configured, the DP communication domain needs to be explicitly initialized
        if int(os.environ.get("VLLM_DP_SIZE", "1")) > 1:
            from recipe.r1_ascend.vllm_parallel_state import init_parallel_state
            init_parallel_state(tensor_parallel_size)
        # NPU-ADAPTATION END

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        VLLM_ENABLE_GRAPGH_MODE = int(os.environ.get("VLLM_ENABLE_GRAPH_MODE", "0"))
        self.inference_engine = LLM(
            model=model_path,
            # NPU-ADAPTATION: Enable inference EP and disable sleep mode.
            enable_sleep_mode=False,
            enable_expert_parallel=True,
            # NPU-ADAPTATION END
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            # NPU-ADAPTATION: Enable graph mode and configure the parameters.
            additional_config={
                "torchair_graph_config": {
                    "enabled": VLLM_ENABLE_GRAPGH_MODE,
                    "use_cached_graph": False,
                    "graph_batch_sizes_init": False,
                    "graph_batch_sizes": [config.max_num_seqs],
                    "enable_multistream_mla": True,
                    "enable_multistream_moe": True,
                    "enable_view_optimize": False,
                    "enable_kv_nz": False,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                },
                "refresh": True,
            },
            # NPU-ADAPTATION END
            **lora_kwargs,
            **engine_kwargs,
        )
        # NPU-ADAPTATION: Weight onload and offload, and initialization configurations such as kv_cache.
        self.model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
        self.kv_cache_configs = None
        self.cpu_model = {}
        self.gpu_buffers = None
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")
        self.free_cache_engine()
        self.offload_model_weights()
        # NPU-ADAPTATION END

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    # NPU-ADAPTATION: Weight onload and offload, kv_cache init and free function
    def init_cache_engine(self):
        if os.environ["VLLM_USE_V1"] == "1":
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            if not worker.model_runner.kv_caches:
                # v1 use explicit initialization method
                self.inference_engine.llm_engine.engine_core.engine_core.model_executor.initialize_from_config(
                    self.inference_engine.llm_engine.engine_core.engine_core.kv_cache_configs
                )
                self.inference_engine.llm_engine.reset_prefix_cache()
        else:
            if self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
                self.inference_engine.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def onload_model_weights(self):
        """
        Advantages over model.cuda():
        1) Avoids CPU to GPU data transfer entirely, leveraging pre-allocated GPU buffers
        instead of copying data from CPU tensors.
        2) Eliminates the recursive traversal of submodules inherent in .cuda(),
        which can be particularly slow for deeply nested model architectures.
        """
        self.gpu_buffers = {}
        for name, param in self.model.named_parameters():
            self.gpu_buffers[name] = torch.empty_like(param, device="cuda")
        for name, param in self.model.named_parameters():
            param.data = self.gpu_buffers[name]

    def offload_model_weights(self):
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]
        if hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                mla = self.model.model.layers[i].self_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None

        self.gpu_buffers = None
        gc.collect()
        torch.npu.empty_cache()

    def free_cache_engine(self):
        if os.environ["VLLM_USE_V1"] == "1":
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context
        else:
            ctx = self.inference_engine.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context
        from vllm.attention import AttentionType

        layer_need_kv_cache = []
        for layer_name in ctx:
            if hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type in (
                AttentionType.DECODER, 
                AttentionType.ENCODER_DECODER,
            ):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = self.inference_engine.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache
        if os.environ["VLLM_USE_V1"] == "1":
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker

            worker.model_runner.kv_caches = []
        else:
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine = None
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.gpu_cache = None

        if hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None

        gc.collect()
        torch.npu.empty_cache()
    # NPU-ADAPTATION END
