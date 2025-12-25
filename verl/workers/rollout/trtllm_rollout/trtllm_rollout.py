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
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Generator, Optional

import aiohttp
import pynvml
import ray
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_torch_device
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 2000
DEFAULT_MAX_WAIT_TIME = 300.0


def get_total_available_bytes(pg: dist.ProcessGroup, rank: int, ratio: float, message: str = "") -> int:
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_free = mem_free + mem_reserved - mem_allocated
    mem_free = torch.tensor(mem_free)
    dist.all_reduce(mem_free, op=dist.ReduceOp.MIN, group=pg)
    mem_free = mem_free.item()
    return int(mem_free * ratio)


def device_id_to_physical_device_id(id: int) -> int:
    """Convert a logical device ID to a physical device ID considering CUDA_VISIBLE_DEVICES."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        try:
            physical_device_id = int(device_ids[id])
            return physical_device_id
        except (ValueError, IndexError) as err:
            raise RuntimeError(
                f"Failed to convert logical device ID {id} to physical device ID. Available devices are: {device_ids}."
            ) from err
    else:
        return id


@contextlib.contextmanager
def nvml_context():
    """Context manager for NVML initialization and shutdown.

    Raises:
        RuntimeError: If NVML initialization fails
    """
    try:
        pynvml.nvmlInit()
        yield
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {e}") from e
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def get_device_uuid(id: int) -> str:
    """Get the UUID of a CUDA device using NVML."""

    # The process has visibility to all GPUs within the TP group
    global_device_idx = device_id_to_physical_device_id(id)

    # Get the device handle and UUID
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(global_device_idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            # Ensure the UUID is returned as a string, not bytes
            if isinstance(uuid, bytes):
                return uuid.decode("utf-8")
            elif isinstance(uuid, str):
                return uuid
            else:
                raise RuntimeError(
                    f"Unexpected UUID type: {type(uuid)} for device {id} (global index: {global_device_idx})"
                )
        except pynvml.NVMLError as e:
            raise RuntimeError(
                f"Failed to get device UUID for device {id} (global index: {global_device_idx}): {e}"
            ) from e


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    if resp.status == 204 or (resp.content_length == 0):
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": (resp.headers.get("Content-Type") or ""),
            "text": text,
        }


class AsyncTRTLLMHttpAdapter:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.max_connections = max_connections

    @asynccontextmanager
    async def _get_session(self) -> aiohttp.ClientSession:
        """Context manager for safe session access with proper connection pooling.

        Yields:
            aiohttp.ClientSession: Session instance for making HTTP requests

        Note:
            This method creates a new session for each request to avoid resource competition
            while still maintaining proper connection pooling through the shared connector.
        """
        # Create a new session for each request to avoid resource competition
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections // 4,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        try:
            yield session
        finally:
            # Always close the session to free up resources
            if not session.closed:
                await session.close()

    async def _make_async_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        method: str = "POST",
        return_status: bool = False,
    ) -> dict[str, Any] | int:
        """Make an async HTTP request with retry logic and consistent error handling.

        Args:
            endpoint (str): The API endpoint to call (without leading slash)
            payload (Optional[Dict[str, Any]], optional): The JSON payload to send.
                Defaults to empty dict if None.
            method (str, optional): HTTP method to use. Defaults to "POST".

        Returns:
            Dict[str, Any]: The JSON response from the server

        Raises:
            aiohttp.ClientResponseError: If the HTTP request fails with a client/server error
            RuntimeError: If all retry attempts are exhausted

        Note:
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """

        url = f"http://{self.host}:{self.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return response.status if return_status else await _read_async_response(response)
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return response.status if return_status else await _read_async_response(response)

            except asyncio.TimeoutError:
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            except aiohttp.ClientConnectorError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts")

    async def resume_memory_occupation(self, tags: list[str]):
        """Resume GPU memory occupation (async version).

        Similar to AsyncEngine, this method handles first-time weight reloading
        by calling release_memory_occupation if needed.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        return await self._make_async_request("resume_memory", {"tags": tags})

    async def release_memory_occupation(self, tags: list[str]):
        """Release GPU memory occupation temporarily (async version).

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return await self._make_async_request("release_memory", {"tags": tags})

    async def update_weights(self, weights: dict[str, str]):
        """Update model weights from tensor data asynchronously.

        Args:
            weights: A dictionary that maps the device uuid of the weight handles.

        Returns:
            Dict[str, Any]: Server response containing update status
        """
        return await self._make_async_request("update_weights", {"weights": weights})


class TRTLLMAsyncRollout(BaseRollout):
    _WEIGHTS_TAGS = [
        "sampler",
        "drafter",
        "guided_decoder",
        "spec_resource_manager",
        "model_extra",
        "executor_extra",
        "model",
        "draft_model",
    ]

    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        super().__init__(config, model_config, device_mesh)
        self._adapter = None
        assert device_mesh.mesh_dim_names.index("dp") == 0, "DP dim should always be the first dimension"

        # Clone a new device mesh for CPU backend only (used for internal ranks communication)
        device_mesh_kwargs = dict(
            mesh_shape=device_mesh.mesh.shape,
            mesh_dim_names=device_mesh.mesh_dim_names,
        )
        self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._device_mesh_cpu[self._device_mesh_cpu.mesh_dim_names[1:]]._flatten(mesh_dim_name="exclude_dp")
        self.is_dp_leader = self._device_mesh_cpu["exclude_dp"].get_local_rank() == 0
        logger.info(f"is_dp_leader: {self.is_dp_leader}")
        logger.info(f"exclude_dp_rank = {self._device_mesh_cpu['exclude_dp'].get_local_rank()}")
        logger.info(f"exclude_dp_size = {self._device_mesh_cpu['exclude_dp'].size()}")

        self.device_mesh = device_mesh
        self.replica_rank = self._device_mesh_cpu["dp"].get_local_rank()
        self.node_rank = 0  # TODO: support multiple nodes
        self.node_ip = ray.util.get_node_ip_address().strip("[]")
        assert len(ray.get_gpu_ids()) == 1, "TRTLLMAsyncRollout should run on a single GPU node"
        self.gpu_id = ray.get_gpu_ids()[0]

    async def _init_server_adapter(self):
        if self._adapter is not None:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"trtllm_server_{self.replica_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        assert server_address == self.node_ip, f"server address: {server_address} != node_ip: {self.node_ip}"

        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._adapter = AsyncTRTLLMHttpAdapter(
            host=host,
            port=server_port,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        if self.is_dp_leader and self.config.free_cache_engine:
            if "weights" in tags:
                tags = self._WEIGHTS_TAGS
            elif "kv_cache" in tags:
                tags = ["kv_cache"]
            else:
                raise ValueError(f"Invalid tag: {tags}")
            await self._init_server_adapter()
            await self._adapter.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.is_dp_leader and self.config.free_cache_engine:
            await self._init_server_adapter()
            tags = self._WEIGHTS_TAGS + ["kv_cache"]
            await self._adapter.release_memory_occupation(tags=tags)

    async def update_weights_from_ipc_handles(self, device_handles):
        """Update weights from IPC handles."""
        if self.is_dp_leader:
            gathered_handles = [None for _ in range(self._device_mesh_cpu["exclude_dp"].size())]
        else:
            gathered_handles = None

        await asyncio.to_thread(
            dist.gather_object,
            obj=device_handles,
            object_gather_list=gathered_handles,
            group_dst=0,
            group=self._device_mesh_cpu["exclude_dp"].get_group(),
        )

        if self.is_dp_leader:
            all_handles = {k: v for d in gathered_handles for k, v in d.items()}
            await self._adapter.update_weights(all_handles)

        await asyncio.to_thread(dist.barrier, group=self._device_mesh_cpu["exclude_dp"].get_group())

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        if self.is_dp_leader:
            await self._init_server_adapter()

        total_available_bytes = await asyncio.to_thread(
            get_total_available_bytes,
            self._device_mesh_cpu["exclude_dp"].get_group(),
            self._device_mesh_cpu["exclude_dp"].get_local_rank(),
            self.config.refit_ipc_memory_ratio,
        )

        try:
            device_uuid = get_device_uuid(self.gpu_id)
        except Exception as e:
            logger.error(f"Failed to get device UUID: {e}")
            logger.error("Did you miss to set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 before ray start?")
            device_uuid = None
            raise e

        cur_available_bytes = total_available_bytes
        cur_handles = []

        async def flush():
            nonlocal cur_available_bytes, cur_handles
            if not cur_handles:
                return
            serialized_device_handles = {device_uuid: base64.b64encode(pickle.dumps(cur_handles)).decode("utf-8")}
            await self.update_weights_from_ipc_handles(serialized_device_handles)
            cur_available_bytes = total_available_bytes
            cur_handles = []

        for name, param in weights:
            size_in_bytes = param.element_size() * param.numel()
            if size_in_bytes > cur_available_bytes:
                await flush()

            assert cur_available_bytes >= size_in_bytes, (
                f"cur_available_bytes: {cur_available_bytes:,} size_in_bytes: {size_in_bytes:,} name: {name}"
            )
            cur_available_bytes -= size_in_bytes
            handle = reduce_tensor(param.detach())
            cur_handles.append((name, handle))

        await flush()

        if self.is_dp_leader:
            # Finalize update weights
            await self._adapter.update_weights(None)
        await asyncio.to_thread(dist.barrier, group=self._device_mesh_cpu["exclude_dp"].get_group())
