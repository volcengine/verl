# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Sharding manager used when DeepSpeed actors feed SGLang rollout workers.

The behaviour largely matches the vLLM equivalent while relying on SGLang's
parallel-state helpers for tensor-parallel collectives.  It preserves RNG state
across context switches and honours ZeRO stage semantics when the actor holds a
partitioned parameter set.
"""

import logging
import os
from typing import Optional

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import check_device_is_available
from verl.workers.sharding_manager.base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DeepSpeedSGLangShardingManager(BaseShardingManager):
    """
    Sharding manager for DeepSpeed + SGLang setup.

    This manager handles:
    1. Data all-gather across SGLang tensor parallel groups
    2. Data chunking for each TP rank after generation
    3. Random state management for reproducibility

    Note: This is similar to FSDP SGLang sharding manager but designed to work
    with DeepSpeed ZeRO optimizer and parameter management.
    """

    @check_device_is_available()
    def __init__(self, inference_engine, device_mesh: Optional[DeviceMesh] = None, zero_stage: int = 2):
        """
        Initialize DeepSpeed SGLang sharding manager.

        Args:
            inference_engine: SGLang inference engine instance
            device_mesh: DeviceMesh defining the parallel topology (optional for single GPU)
            zero_stage: DeepSpeed ZeRO optimization stage (0/1/2/3)
        """
        if inference_engine is None:
            raise ValueError("DeepSpeedSGLangShardingManager requires a valid inference_engine instance.")

        self.device_mesh = device_mesh
        self.inference_engine = inference_engine
        self.zero_stage = zero_stage

        backend_engine = inference_engine
        if not hasattr(backend_engine, "wake_up") and hasattr(backend_engine, "inference_engine"):
            backend_engine = backend_engine.inference_engine
        self._backend_engine = backend_engine

        wake_up_fn = getattr(self._backend_engine, "wake_up", None)
        if callable(wake_up_fn):
            wake_up_fn()

        mesh_names = getattr(device_mesh, "mesh_dim_names", ()) if device_mesh is not None else ()
        if device_mesh is not None and "infer_tp" in mesh_names:
            infer_tp_mesh = device_mesh["infer_tp"]
            self.tp_size = infer_tp_mesh.size()
            self.tp_rank = infer_tp_mesh.get_local_rank()
        else:
            self.tp_size = 1
            self.tp_rank = 0

        if device_mesh is not None and "dp" in mesh_names:
            dp_mesh = device_mesh["dp"]
            self.dp_size = dp_mesh.size()
            dp_rank = dp_mesh.get_local_rank()
        else:
            self.dp_size = 1
            dp_rank = 0
        self.dp_rank = dp_rank

        torch_device = get_torch_device()
        current_rng = torch_device.get_rng_state()
        torch_device.manual_seed(dp_rank + 1000)
        self.gen_random_states = torch_device.get_rng_state()
        torch_device.set_rng_state(current_rng)

        self.timing = {}

        logger.info(
            "DeepSpeedSGLangShardingManager initialized: "
            f"TP={self.tp_size}, TP_rank={self.tp_rank}, DP_rank={dp_rank}, ZeRO_stage={self.zero_stage}"
        )

    @GPUMemoryLogger(role="deepspeed sglang sharding_manager", logger=logger)
    def __enter__(self):
        """Restore the rollout RNG state before weights are pushed."""
        get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="deepspeed sglang sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        """Persist the RNG snapshot and reset any rollout-side caches."""
        self.gen_random_states = get_torch_device().get_rng_state()
        reset_fn = getattr(self._backend_engine, "reset_prefix_cache", None)
        if callable(reset_fn):
            reset_fn()

    @GPUMemoryLogger(role="deepspeed sglang sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        All-gather data across tensor parallel group.

        When using SGLang with TP > 1, each TP rank needs identical input data.
        This method gathers data from all TP ranks so each rank has the full batch.

        Args:
            data: Input DataProto (chunked per TP rank)

        Returns:
            DataProto: All-gathered data (identical across TP ranks)
        """
        if self.tp_size == 1:
            # No TP, no need to gather
            return data

        # Get SGLang tensor parallel process group
        group = sglang_ps.get_tensor_model_parallel_group().device_group

        # All-gather data across TP group
        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="deepspeed sglang sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        Chunk data for this TP rank after generation.

        After SGLang generation, each TP rank has identical output data
        (due to preprocess all-gather). We need to split it back so each
        TP rank only keeps its portion.

        Args:
            data: Generated DataProto (identical across TP ranks)

        Returns:
            DataProto: Chunked data for this TP rank
        """
        if self.tp_size == 1:
            # No TP, no need to chunk
            return data

        # Split data into TP chunks and return this rank's chunk
        return data.chunk(chunks=self.tp_size)[self.tp_rank]
