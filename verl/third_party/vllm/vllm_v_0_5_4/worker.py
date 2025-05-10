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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
"""A GPU worker class."""

import gc
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed
import torch.nn as nn
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoRAConfig,
    MultiModalConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)

# TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
from vllm.distributed import get_tensor_model_parallel_group, init_distributed_environment, set_custom_all_reduce
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest, IntermediateTensors, SamplerOutput
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.model_runner_base import ModelRunnerInputBase
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
from vllm.worker.worker_base import WorkerInput

from .config import LoadConfig, LoadFormat, ModelConfig
from .dtensor_weight_loaders import load_dtensor_weights
from .hf_weight_loader import load_hf_weights
from .megatron_weight_loaders import load_megatron_weights
from .model_runner import ModelRunner
from .parallel_state import ensure_model_parallel_initialized


class Worker(Worker):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],  # model itself or its parameter dict
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        # self.model = model  # will be replaced in the init_model
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.parallel_config.rank = rank
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.prompt_adapter_config = prompt_adapter_config
        self.is_driver_worker = is_driver_worker  # TODO: we don't need driver
        # if parallel_config and is_driver_worker:
        #     assert rank % parallel_config.tensor_parallel_size == 0, \
        #            "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()
        self.multimodal_config = multimodal_config

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_args = {} if speculative_config is None or (speculative_config.draft_model_config.model == model_config.model) or (speculative_config.draft_model_config.hf_config.model_type not in ["medusa", "mlp_speculator"]) else {"return_hidden_states": True}

        # TODO(sgm): set correct model runner class
        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif self.model_config.embedding_mode:
            ModelRunnerClass = EmbeddingModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            model,  # [VERL]: add for verl
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            multimodal_config=multimodal_config,
            **speculative_args,
        )

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine] = None
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None

        # NOTE(sgm): [VERL] For offloading inference engine params
        self.cpu_model = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
            self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
            if self.rank < 0:
                raise ValueError("Invalid or unspecified rank.")
            torch.cuda.set_device(self.device)

            # Use the world_size set by TORCHRUN
            world_size = int(os.getenv("WORLD_SIZE", "-1"))
            assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
            self.parallel_config.world_size = world_size

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method, self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        # self.model = get_model(actor_model=self.model, model_config=self.model_config)

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        assert peak_memory > 0, "Error in memory profiling. This happens when the GPU memory was not properly cleaned up before initializing the vLLM instance."

        cache_block_size = self.get_cache_block_size_bytes()

        # NOTE(sgm) [VERL] use the remaining memory
        num_gpu_blocks = int((free_gpu_memory * self.cache_config.gpu_memory_utilization) // cache_block_size)
        # num_gpu_blocks = int((total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) // cache_block_size)

        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()

        # NOTE(sgm): Add for [VERL], synchronize number of blocks with all the rank
        num_gpu_blocks = torch.tensor([num_gpu_blocks], device="cuda")
        num_cpu_blocks = torch.tensor([num_cpu_blocks], device="cuda")

        torch.distributed.all_reduce(num_gpu_blocks, op=torch.distributed.ReduceOp.MIN, group=get_tensor_model_parallel_group().device_group)
        torch.distributed.all_reduce(num_cpu_blocks, op=torch.distributed.ReduceOp.MIN, group=get_tensor_model_parallel_group().device_group)
        num_gpu_blocks = num_gpu_blocks.item()
        num_cpu_blocks = num_cpu_blocks.item()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def _init_cache_engine(self):
        if self.cache_engine is None and self.gpu_cache is None:
            super()._init_cache_engine()

    def free_cache_engine(self):
        # ensure `enforce_eager=True`
        self.cache_engine = None
        self.gpu_cache = None

    # NOTE(sgm): [VERL]: adapt from _execute_model_spmd()
    def execute_model(self, execute_model_req: ExecuteModelRequest, intermediate_tensors: Optional[IntermediateTensors] = None) -> Optional[List[SamplerOutput]]:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, "_execute_model_spmd() requires each worker to take in an ExecuteModelRequest"
        worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = self.model_runner.prepare_model_input(execute_model_req.seq_group_metadata_list)

        # verl.worker.workerbase.WorkerBase
        # swap cache
        super().execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        return self.model_runner.execute_model(
            model_input,
            self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
            intermediate_tensors,
        )

    # assume the input is .state_dict()
    def sync_model_weights(self, actor_weights: Dict, load_format: str):
        if load_format in [LoadFormat.MEGATRON, LoadFormat.AUTO]:
            load_megatron_weights(actor_weights, self.model_runner.model)
        elif load_format == LoadFormat.HF:
            # full model state dict without no sharding
            load_hf_weights(actor_weights, self.model_runner.model)
        elif load_format == LoadFormat.DTENSOR:
            load_dtensor_weights(actor_weights, self.model_runner.model)

    def offload_model_weights(self) -> None:
        if self.cpu_model is None:
            self.cpu_model = {}
            for name, params in self.model_runner.model.named_parameters():
                self.cpu_model[name] = torch.empty_like(params, device="cpu")
                params.data = self.cpu_model[name]
        else:
            for name, params in self.model_runner.model.named_parameters():
                params.data = self.cpu_model[name]


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = "env://",
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # NOTE(sgm) use tcp://localhost:xxxx will hang in HF setting without megatron
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank)

    ensure_model_parallel_initialized(
        tensor_model_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
    )

    # TODO(sgm): check whether need this
    # if pynccl_utils.is_initialized():
    #     pynccl_world_size = pynccl_utils.get_world_size()
    #     if pynccl_world_size != parallel_config.world_size:
    #         raise RuntimeError(
    #             "pynccl is already initialized but the pynccl world "
    #             "size does not match parallel_config.world_size "
    #             f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    # elif parallel_config.world_size > 1:
    #     # NOTE(woosuk): We don't initialize pynccl process group when world size
    #     # is 1.
    #     # NOTE(kaichao): By default, pynccl is initialized for tp group.
    #     pynccl_utils.init_process_group(
    #         group=get_tensor_model_parallel_cpu_group())

    # # Initialize a custom fast all-reduce implementation.
    # if not parallel_config.disable_custom_all_reduce:
    #     init_custom_ar()

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # if pynccl_utils.is_initialized():
    #     pynccl_utils.all_reduce(torch.zeros(1).cuda())
