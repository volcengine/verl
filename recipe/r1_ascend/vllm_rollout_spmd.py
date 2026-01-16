# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
# Adapted from
# https://github.com/volcengine/verl/blob/main/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py

import logging
import os
from typing import Generator

import torch
import torch.distributed
from omegaconf import ListConfig
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel

from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import get_device_name
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout import vLLMRollout as vLLMRolloutBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMRollout(vLLMRolloutBase):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh
        # NPU-ADAPTATION: import vLLM-Ascend patch
        from vllm_ascend.patch import (
            platform,  # noqa: F401
            worker,  # noqa: F401
        )

        from recipe.r1_ascend import engine_core  # noqa: F401
        # NPU-ADAPTATION END

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

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

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

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
                    "enable_multistream_mla": False,
                    "enable_multistream_moe": False,
                    "enable_view_optimize": False,
                    "enable_kv_nz": False,
                    "enable_frozen_parameter": False,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                },
                "refresh": True,
            },
            # NPU-ADAPTATION END
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )
        # NPU-ADAPTATION: Weight onload and offload, and initialization configurations such as kv_cache.
        self.model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
        self.kv_cache_configs = None
        self.cpu_model = {}
        self.gpu_buffers = None
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")
        # NPU-ADAPTATION END

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        logger.info(f"vllm sampling kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    # NPU-ADAPTATION: Weight onload and offload, kv_cache init and free function
    # NOTE: Due to potential incomplete memory offloading during sleep operations for vLLM on NPUs, we add
    # patches to manually handle the off/on loading of the rollout model and KVcache on NPUs.
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
        self.gpu_buffers = {}
        for name, param in self.model.named_parameters():
            self.gpu_buffers[name] = torch.empty_like(param, device=get_device_name())
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
        aggressive_empty_cache()

    def free_cache_engine(self):
        if os.environ["VLLM_USE_V1"] == "1":
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context
        else:
            compilation_config = self.inference_engine.llm_engine.model_executor.driver_worker.worker.compilation_config
            ctx = compilation_config.static_forward_context
        from vllm.attention import AttentionType

        layer_need_kv_cache = []
        for layer_name in ctx:
            if hasattr(ctx[layer_name], "attn_type") and ctx[layer_name].attn_type in (
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

        aggressive_empty_cache()

    def _process_mla(self, load_weight=False):
        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            mla = self.model.model.layers[i].self_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
            if load_weight:
                mla.process_weights_after_loading(None)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in NPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "weights" in tags:
            self.onload_model_weights()
        elif "kv_cache" in tags:
            self.init_cache_engine()

    async def release(self):
        """Release weights and kv cache in NPU memory."""
        if not self.config.free_cache_engine:
            return

        self.free_cache_engine()
        self.offload_model_weights()

        if hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            self._process_mla()

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        await super().update_weights(weights, **kwargs)

        if hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            self._process_mla(load_weight=True)

    # NPU-ADAPTATION END
