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
import os
import gc
from typing import Dict, List, Tuple, Optional, Union, Set

import torch
import torch.distributed
import torch.nn as nn

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (initialize_model_parallel)
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group

from .model_runner import ModelRunner
from .model_loader import load_weights
from .parallel_state import initialize_model_parallel_from_megatron
from vllm.lora.request import LoRARequest


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
    ) -> None:
        # self.model = model  # will be replaced in the init_model
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config

        self.model_runner = ModelRunner(
            model,
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            lora_config=self.lora_config,
            kv_cache_dtype=kv_cache_dtype,
        )

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.sliding_window = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

        # For offloading inference engine params
        self.cpu_model = None

    def init_model(self, cupy_port: Optional[int] = None):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # Env vars will be set by TORCHRUN.
        self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        _check_if_gpu_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        # TODO: do not use cupy
        _init_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method)
        if not self.parallel_config.disable_custom_all_reduce:
            init_custom_ar()
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        # self.model = get_model(actor_model=self.model, model_config=self.model_config)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
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

        cache_block_size = CacheEngine.get_cache_block_size(block_size, cache_dtype, self.model_config,
                                                            self.parallel_config)
        # NOTE(sgm) use the remaining memory
        num_gpu_blocks = int((free_gpu_memory * gpu_memory_utilization) // cache_block_size)
        # num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        # Synchronize number of blocks with all the rank
        num_gpu_blocks = torch.tensor([num_gpu_blocks], device='cuda')
        num_cpu_blocks = torch.tensor([num_cpu_blocks], device='cuda')
        torch.distributed.all_reduce(num_gpu_blocks,
                                     op=torch.distributed.ReduceOp.MIN,
                                     group=get_tensor_model_parallel_group())
        torch.distributed.all_reduce(num_cpu_blocks,
                                     op=torch.distributed.ReduceOp.MIN,
                                     group=get_tensor_model_parallel_group())
        num_gpu_blocks = num_gpu_blocks.item()
        num_cpu_blocks = num_cpu_blocks.item()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        if self.cache_engine is None and self.gpu_cache is None:
            self.cache_config = cache_config
            self.cache_engine = CacheEngine(self.cache_config, self.model_config, self.parallel_config)
            self.cache_events = self.cache_engine.events
            self.gpu_cache = self.cache_engine.gpu_cache
            self.model_runner.set_block_size(self.cache_engine.block_size)

    def free_cache_engine(self):
        # ensure `enforce_eager=True`
        self.cache_engine = None
        self.gpu_cache = None

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        num_seq_groups = len(seq_group_metadata_list)
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}
        output = self.model_runner.execute_model(seq_group_metadata_list, self.gpu_cache)
        return output

        # # Prepare input tensors.
        # # NOTE(shengguangming): currently we pad in our dataloader and unpad it in pre_process_input, j
        # # we can just input un-padded sequence for better performance
        # input_tokens, input_positions, input_metadata = self._prepare_inputs(seq_group_metadata_list)

        # # Execute the model.
        # output = self.model(
        #     input_ids=input_tokens,
        #     positions=input_positions,
        #     kv_caches=self.gpu_cache,
        #     input_metadata=input_metadata,
        #     cache_events=cache_events,
        # )
        # return output

    # assume the input is .state_dict()
    def sync_model_weights(self, actor_weights: Dict):
        load_weights(actor_weights, self.model_runner.model)

    def offload_model_weights(self) -> None:
        if self.cpu_model == None:
            self.cpu_model = {}
            for name, params in self.model_runner.model.named_parameters():
                self.cpu_model[name] = torch.empty_like(params, device='cpu')
                params.data = self.cpu_model[name]
        else:
            for name, params in self.model_runner.model.named_parameters():
                params.data = self.cpu_model[name]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        print('The distributed environment has been initialized before vLLM')
    elif not distributed_init_method:
        raise ValueError("distributed_init_method must be set if torch.distributed "
                         "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            # init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # TODO (shengguangming): maybe we should also flag the megatron is initialized
    if torch.distributed.get_world_size() > 1:
        initialize_model_parallel_from_megatron(tensor_model_parallel_size=parallel_config.tensor_parallel_size)
    else:
        initialize_model_parallel()


def _pad_to_alignment(x: List[int], multiple_of: int, pad: int) -> List[int]:
    return x + [pad] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    return x + [pad] * (max_len - len(x))


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError("Bfloat16 is only supported on GPUs with compute capability "
                             f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                             f"{compute_capability[0]}.{compute_capability[1]}.")
