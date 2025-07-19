# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import asyncio
import logging
import os
from typing import Iterator

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor, FlattenedTensorBucket, FlattenedTensorMetadata
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.torch_functional import check_device_is_available
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def get_flattened_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int
) -> Iterator[FlattenedTensorBucket]:
    """
    Group tensors into flattened buckets based on a specified size in bytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        FlattenedTensorBucket objects containing flattened tensors with metadata.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_flattened_tensor_buckets(tensors, bucket_bytes=10*1024*1024):
        ...     print(bucket)
        FlattenedTensorBucket(num_tensors=2, num_elements=5000000, size=20.00MB)
    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0

    for name, tensor in iterable:
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield FlattenedTensorBucket(named_tensors=current_bucket)
            current_bucket = [(name, tensor)]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor))
            current_size += tensor_size

    if current_bucket:
        yield FlattenedTensorBucket(named_tensors=current_bucket)


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        multi_stage_wake_up: bool = False,
        use_flattened_buckets: bool = True,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.multi_stage_wake_up = multi_stage_wake_up
        self.use_flattened_buckets = use_flattened_buckets

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        named_tensors = [(k, v) for k, v in params.items()]
        # convert megabytes to bytes
        update_weights_bucket_bytes = int(self.rollout_config.update_weights_bucket_megabytes) << 20

        print(f"Hello Biao: update_weights_bucket_bytes is {update_weights_bucket_bytes}")
        print(f"Hello Biao: use_flattened_buckets is {self.use_flattened_buckets}")
        
        if self.use_flattened_buckets:
            await self._update_weights_by_flattened_bucket(named_tensors, update_weights_bucket_bytes)
        else:
            await self._update_weights_by_bucket(named_tensors, update_weights_bucket_bytes)

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self.inference_engine.flush_cache()

    async def _update_weights_by_bucket(self, named_tensors, update_weights_bucket_bytes):
        """Original bucket-based weight update method"""
        load_format = None
        
        for batch in get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            # On each rank, serialize a batch of (name, tensor) tuples.
            # named_tensors_batch will be a list like:
            # [(name0, serialized_tensor0_tp0), (name1, serialized_tensor1_tp0), ...]
            named_tensors_batch = [
                (name, MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor)))
                for name, tensor in batch
            ]

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # On rank 0, prepare a list to hold the gathered batches from all ranks.
                gathered_serialized_batches = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_batches = None

            # Gather the named_tensors_batch from all ranks to rank 0.
            # After this, on rank 0, gathered_serialized_batches will be a list of lists:
            # [ [ (name0, s_t0_tp0), (name1, s_t1_tp0), ... ],  # batch from TP rank 0
            #   [ (name0, s_t0_tp1), (name1, s_t1_tp1), ... ],  # batch from TP rank 1
            #   ... ]
            # On other ranks, gathered_serialized_batches will be None.
            dist.gather_object(
                obj=named_tensors_batch,
                object_gather_list=gathered_serialized_batches,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # Use zip(*) to "transpose" the data structure.
                # This groups the serialized parts for each individual tensor across all TP ranks.
                # Example: from [[(n0, t0_tp0), (n1, t1_tp0)], [(n0, t0_tp1), (n1, t1_tp1)]]
                # to [ ( (n0, t0_tp0), (n0, t0_tp1) ), ( (n1, t1_tp0), (n1, t1_tp1) ) ]
                logical_tensors = zip(*gathered_serialized_batches, strict=True)

                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        # 'tensor_group' represents a single logical tensor's data from all ranks.
                        (
                            tensor_group[0][0],  # Get the name from the first rank's data.
                            LocalSerializedTensor(
                                # 'rank_part' is the (name, serialized_tensor) tuple from one specific rank.
                                values=[rank_part[1] for rank_part in tensor_group]
                            ),
                        )
                        for tensor_group in logical_tensors
                        # each tensor_group is like ( (n0, t0_tp0), (n0, t0_tp1) )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )

    async def _update_weights_by_flattened_bucket(self, named_tensors, update_weights_bucket_bytes):
        """Flattened bucket-based weight update method for better memory efficiency"""
        total_buckets = 0
        
        for flattened_bucket in get_flattened_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            # Get the flattened tensor and metadata
            flattened_tensor = flattened_bucket.get_flattened_tensor()
            metadata = flattened_bucket.get_metadata()

            # Preprocess the flattened tensor
            processed_tensor = _preprocess_tensor_for_update_weights(flattened_tensor)
            
            # Serialize the flattened tensor and metadata
            serialized_flattened = MultiprocessingSerializer.serialize(processed_tensor.detach())
            serialized_data = {
                "flattened_tensor": serialized_flattened,
                "metadata": metadata,
            }

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                gathered_serialized_data = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_data = None

            dist.gather_object(
                obj=serialized_data,
                object_gather_list=gathered_serialized_data,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # Create a special entry for the flattened bucket
                bucket_name = f"__flattened_bucket_{total_buckets}"

                # Prepare the flattened tensor data with metadata
                flattened_data = {
                    "serialized_tensors": [
                        rank_data["flattened_tensor"]
                        for rank_data in gathered_serialized_data
                    ],
                    "metadata": [
                        rank_data["metadata"] for rank_data in gathered_serialized_data
                    ],
                    "bucket_id": total_buckets,
                }

                # Serialize the flattened data
                serialized_flattened_data = MultiprocessingSerializer.serialize(flattened_data)

                # Create LocalSerializedTensor with entries for all ranks
                # Each rank gets the same serialized data since rank 0 handles all processing
                world_size = self.device_mesh["infer_tp"].mesh.size()[0]
                serialized_values = [serialized_flattened_data for _ in range(world_size)]

                # Use the existing update_weights_from_tensor method with a special format
                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (bucket_name, LocalSerializedTensor(values=serialized_values))
                    ],
                    load_format="flattened_bucket",
                    flush_cache=False,
                )

            total_buckets += 1

    async def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        get_torch_device().empty_cache()

        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.resume_memory_occupation(tags=["weights"])
                log_gpu_memory_usage("Before resume SGLang weights in sharding manager", logger=logger)
            else:
                await self.inference_engine.resume_memory_occupation()
                log_gpu_memory_usage("Before resume SGLang weights + kv_cache in sharding manager", logger=logger)

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        params = {
            k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()
        }

        # convert weight keys to match the model config
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        # Copy, not share memory
        await self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        if (
            self.multi_stage_wake_up
            and self.rollout_config.free_cache_engine
            and self.device_mesh["infer_tp"].get_local_rank() == 0
        ):
            await self.inference_engine.resume_memory_occupation(tags=["kv_cache"])
            log_gpu_memory_usage("After resume SGLang kv_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
