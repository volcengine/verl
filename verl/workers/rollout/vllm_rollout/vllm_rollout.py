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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import logging
import os
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator, Optional

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from torch.distributed.device_mesh import DeviceMesh
from torch.multiprocessing.reductions import reduce_tensor
from vllm.config import LoRAConfig

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class ServerAdapter(BaseRollout):
    """
    vLLM server adapter used in native async mode, serve as a client to request vLLM server
    to resume/release/update weights and kv_cache.
    """
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size 
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.rollout_rank = rank % rollout_world_size
        self.local_rank = self.rollout_rank % local_world_size

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL
        
        # Attributes related to weight updates
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(0)
        self.zmq_context = zmq.Context()
        self.zmq_address_counter = 0
        self.zmq_handles: dict[str, str] = None

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if self.rollout_rank != 0:
            return None

        if not hasattr(self, "server_handle") or self.server_handle is None:
            raise RuntimeError("vLLMHttpServer handle not set")

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            await self._execute_method("wake_up", kwargs={"tags": tags})

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            await self._execute_method("sleep", kwargs={"level": self.sleep_level})

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update model weights via CUDA IPC to inference workers."""
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            await self._execute_method(
                "update_lora_weights_from_ipc",
                non_block=True,
                kwargs={
                    "peft_config": peft_config,
                    "zmq_handles": self.zmq_handles,
                }
            )
        else:
            await self._execute_method(
                "update_weights_from_ipc",
                non_block=True,
                kwargs={
                    "zmq_handles": self.zmq_handles,
                }
            )
        await self._update_weights_per_tensor(weights)

    async def _update_weights_per_tensor(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Update weights per tensor via ZMQ."""
        s = self.zmq_context.socket(zmq.REQ)
        s.bind(self.zmq_address)

        for name, p in weights:
            handle = reduce_tensor(p)
            # Send the tensor handle
            s.send_pyobj(handle)
            s.recv()
            # Send metadata (tensor name)
            s.send_pyobj({"name": name})
            s.recv()

        # Send None to signal completion
        s.send_pyobj(None)
        s.recv()
        s.close()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def set_server_handle(self, server_handle: ray.actor.ActorHandle):
        """Set vLLMHttpServer handle"""
        if self.rollout_rank == 0:
            self.server_handle = server_handle
    
    def get_update_weights_zmq_handle(self) -> dict[str, str]:
        """Get ZMQ handle for weight updates."""
        suffix = f"{self.device_uuid}-{self.zmq_address_counter}"
        self.zmq_address = f"ipc:///tmp/rl-colocate-zmq-{suffix}.sock"
        self.zmq_address_counter += 1
        return {self.device_uuid: self.zmq_address}

    def set_update_weights_zmq_handles(self, zmq_handles: dict[str, str]):
        """Set ZMQ handles for all workers."""
        self.zmq_handles = zmq_handles
