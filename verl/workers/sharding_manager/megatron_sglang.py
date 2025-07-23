# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

import asyncio
import logging
import os

import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
    per_tensor_generator,
)
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all 
  the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


class MegatronSGLangShardingManager(BaseShardingManager):
    """A sharding manager for Megatron-style training & inference with SGLang.

    This class manages the sharding of model parameters between training and inference
    phases in a Megatron-style parallel setup. It handles:
    - Loading/offloading parameters between CPU/GPU
    - Updating inference engine weights
    - Managing random states for reproducibility
    - Data preprocessing for distributed inference

    Args:
        actor_module (nn.ModuleList): The actor model modules
        inference_engine (Engine): The SGLang inference engine
        model_config: Configuration for the actor's model
        rollout_config: Configuration for rollout generation
        transformer_config: Transformer-specific configuration
        layer_name_mapping: Mapping between layer names and parameters
        weight_converter: Utility for converting weights between formats
        device_mesh (DeviceMesh | None): PyTorch device mesh for distributed training
        offload_param (bool): Whether to offload parameters to CPU when not in use
    """

    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config: DictConfig,
        rollout_config: DictConfig,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
        offload_param: bool = False,
        bridge=None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.device_mesh = device_mesh
        self.bridge = bridge
        self.offload_param = offload_param

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size

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

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        if self.device_mesh["tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.resume_memory_occupation()
        named_tensors = params
        load_format = None

        update_weights_bucket_bytes = int(self.rollout_config.update_weights_bucket_megabytes) << 20
        for batch in get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            # On each rank, serialize a batch of (name, tensor) tuples.
            # named_tensors_batch will be a list like:
            # [(name0, serialized_tensor0_tp0), (name1, serialized_tensor1_tp0), ...]
            named_tensors_batch = [
                (name, MultiprocessingSerializer.serialize(tensor.detach())) for name, tensor in batch
            ]

            if self.device_mesh["tp"].get_local_rank() == 0:
                # On rank 0, prepare a list to hold the gathered batches from all ranks.
                gathered_serialized_batches = [None for _ in range(self.device_mesh["tp"].mesh.size()[0])]
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
                dst=self.device_mesh["tp"].mesh.tolist()[0],
                group=self.device_mesh["tp"].get_group(),
            )

            if self.device_mesh["tp"].get_local_rank() == 0:
                # Use zip(*) to "transpose" the data structure.
                # This groups the serialized parts for each individual tensor across all TP ranks.
                # Example: from [[(n0, t0_tp0), (n1, t1_tp0)], [(n0, t0_tp1), (n1, t1_tp1)]]
                # to [ ( (n0, t0_tp0), (n0, t0_tp1) ), ( (n1, t1_tp0), (n1, t1_tp1) ) ]
                logical_tensors = zip(*gathered_serialized_batches, strict=False)
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

        if self.device_mesh["tp"].get_local_rank() == 0:
            await self.inference_engine.flush_cache()

    async def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        if self.offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self.bridge is not None:
            per_tensor_param = self.bridge.export_weights(self.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor_module,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
            )
        await self.update_weights(per_tensor_param)
        if self.offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        get_torch_device().empty_cache()
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.device_mesh["tp"].get_group())
        return data

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.device_mesh["tp"].get_local_rank()]
