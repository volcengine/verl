# Copyright 2025 Zhipu AI
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
#
# This file is adapted from multiple sources:
# 1. THUDM/slime project
#    Original source: https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/http_server_engine.py
#    Copyright 2025 Zhipu AI
#    Licensed under the Apache License, Version 2.0
# 2. SGLang project
#    Original source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server_engine.py
#    Copyright 2023-2024 SGLang Team
#    Licensed under the Apache License, Version 2.0
#
# Modifications made by Zhipu AI and ModelBest Inc. include but are not limited to:
# - Enhanced error handling and retry logic
# - Added async support with connection pooling
# - Extended functionality for distributed weight updates
# - Improved logging and monitoring capabilities
# - Additional configuration options and optimizations

"""HTTP Server Engine Adapter for SGLang.

This module provides HTTP-based adapters for SGLang engines, allowing communication
with SGLang servers through HTTP requests instead of direct engine calls.

Classes:
    HttpServerEngineAdapter: Synchronous HTTP adapter for SGLang engines
    AsyncHttpServerEngineAdapter: Asynchronous HTTP adapter for SGLang engines

Functions:
    launch_server_process: Launch and initialize an SGLang HTTP server process
"""

import asyncio
import logging
import multiprocessing
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import requests
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration constants
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 100


def launch_server_process(server_args: ServerArgs, timeout: float = DEFAULT_TIMEOUT) -> multiprocessing.Process:
    """Launch an SGLang HTTP server process and wait for it to be ready.

    This function starts a new process running an SGLang HTTP server, then waits
    for the server to become ready by polling its health endpoints. It ensures
    the server is fully operational before returning.

    Args:
        server_args (ServerArgs): Server configuration arguments including host, port, and other settings
        timeout (float, optional): Timeout for individual HTTP requests during health checks.
            Defaults to DEFAULT_TIMEOUT.

    Returns:
        multiprocessing.Process: The launched multiprocessing.Process instance

    Raises:
        RuntimeError: If the server process terminates unexpectedly during startup or cache flush
        TimeoutError: If server fails to become ready within reasonable time (300 seconds)
        requests.RequestException: If health check requests fail repeatedly

    Note:
        This function will return immediately for non-master nodes (node_rank != 0),
        but the process will still be started and returned.
    """
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return p

    base_url = server_args.url()
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {server_args.api_key}",
    }

    # Health check with overall timeout
    start_time = time.time()
    max_wait_time = 300.0  # 5 minutes max wait

    with requests.Session() as session:
        while time.time() - start_time < max_wait_time:
            if not p.is_alive():
                raise RuntimeError("Server process terminated unexpectedly during startup")

            try:
                response = session.get(f"{base_url}/health_generate", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(2)
        else:
            p.terminate()
            raise TimeoutError("Server failed to become healthy within timeout period")

        # Ensure cache is ready
        while time.time() - start_time < max_wait_time:
            if not p.is_alive():
                raise RuntimeError("Server process terminated unexpectedly during cache flush")

            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Cache flush check failed: {e}")

            time.sleep(2)
        else:
            p.terminate()
            raise TimeoutError("Server cache flush failed within timeout period")

    return p


class HttpServerEngineAdapter(EngineBase):
    """HTTP-based adapter for SGLang engines.

    This adapter allows interaction with SGLang engines through HTTP requests
    instead of direct engine calls. It launches an HTTP server process and
    provides methods to communicate with it via REST API calls.

    You can use this class to launch a server from a HttpServerEngineAdapter instance.
    We recommend using this class only when you need to use http server.
    Otherwise, you can use Engine directly.

    Attributes:
        router_ip (Optional[str]): IP address of the router for worker registration
        router_port (Optional[int]): Port of the router for worker registration
        server_args (ServerArgs): Server configuration arguments
        node_rank (int): Rank of this node in distributed setup
        process (multiprocessing.Process): The launched server process
        timeout (float): HTTP request timeout in seconds
        max_retries (int): Maximum number of retry attempts for failed requests
        retry_delay (float): Base delay between retries in seconds
    """

    def __init__(
        self,
        router_ip: Optional[str] = None,
        router_port: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        **kwargs: Any,
    ) -> None:
        """Initialize the HTTP server engine adapter.

        Args:
            router_ip (Optional[str], optional): IP address of router for worker registration.
                Defaults to None.
            router_port (Optional[int], optional): Port of router for worker registration.
                Defaults to None.
            timeout (float, optional): HTTP request timeout in seconds.
                Defaults to DEFAULT_TIMEOUT.
            max_retries (int, optional): Maximum number of retry attempts for failed requests.
                Defaults to DEFAULT_MAX_RETRIES.
            retry_delay (float, optional): Base delay between retries in seconds.
                Defaults to DEFAULT_RETRY_DELAY.
            **kwargs (Any): Additional arguments passed to ServerArgs

        Note:
            If both router_ip and router_port are provided and this is the master node
            (node_rank == 0), the adapter will automatically register with the router.
        """
        self.router_ip: Optional[str] = router_ip
        self.router_port: Optional[int] = router_port
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        self.server_args: ServerArgs = ServerArgs(**kwargs)
        self.node_rank: int = self.server_args.node_rank

        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}")
        self.process: multiprocessing.Process = launch_server_process(self.server_args, self.timeout)

        if self.node_rank == 0 and self.router_ip and self.router_port:
            self._register_with_router()

    def _register_with_router(self) -> None:
        """Register worker with router with error handling.

        This method attempts to register the current worker with a router service.
        If registration fails, it logs an error but does not raise an exception,
        allowing the server to continue operating without router integration.

        Raises:
            Does not raise exceptions - all errors are logged and handled gracefully.
        """
        try:
            url = f"http://{self.router_ip}:{self.router_port}/add_worker"
            params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
            response = requests.post(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            logger.info("Successfully registered with router")
        except Exception as e:
            logger.error(f"Failed to register with router: {e}")
            # Don't raise here - server can still work without router

    def _make_request(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        only_master: bool = True,
    ) -> Dict[str, Any]:
        """Make a HTTP request with retry logic and consistent error handling.

        Args:
            endpoint (str): The API endpoint to call (without leading slash)
            payload (Optional[Dict[str, Any]], optional): The JSON payload to send.
                Defaults to empty dict if None.
            method (str, optional): HTTP method to use. Defaults to "POST".

        Returns:
            Dict[str, Any]: The JSON response from the server

        Raises:
            requests.HTTPError: If the HTTP request fails with a client/server error
            RuntimeError: If all retry attempts are exhausted

        Note:
            - For non-master nodes (node_rank != 0), returns empty dict immediately
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, timeout=self.timeout)
                else:
                    response = requests.post(url, json=payload or {}, timeout=self.timeout)

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Request to {endpoint} timed out (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_retries:
                    raise

            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff

        raise RuntimeError(f"Failed to complete request to {endpoint} after {self.max_retries + 1} attempts")

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ) -> Dict[str, Any]:
        """Update model weights from tensor data.

        The HTTP server will only post meta data, and the real weights will be
        copied directly from GPUs.

        Args:
            serialized_named_tensors (List[str]): List of serialized tensor data
            load_format (Optional[str], optional): Format specification for loading weights.
                Defaults to None.
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to False.

        Returns:
            Dict[str, Any]: Server response containing update status

        Note:
            The model should be on GPUs rather than CPU for this functionality to work properly.
            If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def shutdown(self) -> None:
        """Shutdown the HTTP server and clean up resources.

        This method performs the following cleanup operations:
        1. Unregisters the worker from the router (if configured)
        2. Terminates the server process tree

        All operations are performed with error handling to ensure graceful shutdown
        even if individual steps fail.

        Note:
            This method should be called when the adapter is no longer needed
            to ensure proper cleanup of resources and processes.
        """
        # Unregister from router
        if self.router_ip and self.router_port:
            try:
                url = f"http://{self.router_ip}:{self.router_port}/remove_worker"
                params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
                requests.post(url, params=params, timeout=5.0)  # Short timeout for shutdown
                logger.info("Successfully unregistered from router")
            except Exception as e:
                logger.warning(f"Failed to unregister from router: {e}")

        # Kill server process
        if hasattr(self, "process") and self.process is not None:
            try:
                kill_process_tree(self.process.pid)
                logger.info("Server process terminated")
            except Exception as e:
                logger.error(f"Failed to terminate server process: {e}")

    def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        input_ids: Optional[List[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate text using the SGLang server.

        Args:
            prompt (Optional[str], optional): Text prompt for generation. Defaults to None.
            sampling_params (Optional[Dict[str, Any]], optional): Parameters controlling
                text generation sampling. Defaults to None.
            input_ids (Optional[List[int]], optional): Alternative to prompt, direct token IDs input.
                Defaults to None.
            image_data (Optional[Any], optional): Image data for multimodal generation.
                Defaults to None.
            return_logprob (bool, optional): Whether to return log probabilities.
                Defaults to False.
            logprob_start_len (Optional[int], optional): Starting length for log probability calculation.
                Defaults to None.
            top_logprobs_num (Optional[int], optional): Number of top log probabilities to return.
                Defaults to None.
            token_ids_logprob (Optional[List[int]], optional): Specific token IDs for
                log probability calculation. Defaults to None.
            lora_path (Optional[str], optional): Path to LoRA adapter weights. Defaults to None.
            custom_logit_processor (Optional[Callable], optional): Custom logit processing function.
                Defaults to None.

        Returns:
            Dict[str, Any]: Generated text and associated metadata from the server

        Note:
            Either prompt or input_ids should be provided, but not both.
            The response format depends on the server configuration and parameters.
        """
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("generate", payload, only_master=False)

    def flush_cache(self) -> Dict[str, Any]:
        """Flush the cache of the server.

        This method repeatedly attempts to flush the server cache until successful.
        The flush operation will not return status 200 when there are pending requests.

        Returns:
            Dict[str, Any]: Server response indicating cache flush status.
                For non-master nodes, returns empty dict.

        Note:
            Uses retry logic with limited attempts (max_retries * 2) to avoid infinite loops.
            Each retry includes a delay to allow pending requests to complete.
        """
        if self.node_rank != 0:
            return {}

        # Use retry logic with limited attempts to avoid infinite loops
        for attempt in range(self.max_retries * 2):  # Allow more retries for cache flush
            try:
                response = requests.get(
                    f"http://{self.server_args.host}:{self.server_args.port}/flush_cache", timeout=self.timeout
                )
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

            time.sleep(self.retry_delay)

        logger.error("Failed to flush cache after maximum attempts")
        return {}

    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Release GPU memory occupation temporarily.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return self._make_request("release_memory_occupation", {"tags": tags})

    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Resume GPU memory occupation.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        return self._make_request("resume_memory_occupation", {"tags": tags})

    def init_weights_update_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int, group_name: str, backend: str
    ) -> Dict[str, Any]:
        """Initialize a distributed weights update group.

        Args:
            master_address (str): Address of the master node for distributed communication
            master_port (int): Port of the master node
            rank_offset (int): Offset for process ranks in the group
            world_size (int): Total number of processes in the distributed group
            group_name (str): Name identifier for the process group
            backend (str): Backend to use for distributed communication (e.g., 'nccl', 'gloo')

        Returns:
            Dict[str, Any]: Server response indicating group initialization status
        """
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def update_weights_from_distributed(
        self,
        names: List[str],
        dtypes: List[Any],
        shapes: List[Tuple[int, ...]],
        group_name: str,
        flush_cache: bool = False,
    ) -> Dict[str, Any]:
        """Update model weights from distributed tensors.

        Args:
            names (List[str]): List of tensor names to update
            dtypes (List[Any]): List of data types for each tensor (typically torch.dtype)
            shapes (List[Tuple[int, ...]]): List of tensor shapes
            group_name (str): Name of the distributed process group
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to False.

        Returns:
            Dict[str, Any]: Server response indicating distributed update status
        """
        return self._make_request(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
                "flush_cache": flush_cache,
            },
        )

    def pause_generation(self) -> Dict[str, Any]:
        """Pause text generation on the server.

        Returns:
            Dict[str, Any]: Server response indicating pause status
        """
        return self._make_request("pause_generation", {})

    def continue_generation(self) -> Dict[str, Any]:
        """Continue text generation on the server.

        Returns:
            Dict[str, Any]: Server response indicating continuation status
        """
        return self._make_request("continue_generation", {})


class AsyncHttpServerEngineAdapter(HttpServerEngineAdapter):
    """Asynchronous HTTP-based adapter for SGLang engines.

    This class inherits from HttpServerEngineAdapter and adds async capabilities
    for non-blocking HTTP requests to the SGLang server. It provides the same
    functionality as the synchronous version but with async/await support.

    The async adapter is useful when you need to make multiple concurrent requests
    or integrate with async frameworks. It uses aiohttp for efficient async HTTP
    communication and maintains connection pooling for better performance.

    Attributes:
        _need_reload (bool): Flag indicating if weights need to be reloaded on first use
        _session (Optional[aiohttp.ClientSession]): aiohttp ClientSession for making async HTTP requests
        _session_lock (asyncio.Lock): Lock for thread-safe session access
        max_connections (int): Maximum number of connections in the connection pool
    """

    def __init__(
        self,
        router_ip: Optional[str] = None,
        router_port: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        **kwargs: Any,
    ) -> None:
        """Initialize the async HTTP server engine adapter.

        Args:
            router_ip (Optional[str], optional): IP address of router for worker registration.
                Defaults to None.
            router_port (Optional[int], optional): Port of router for worker registration.
                Defaults to None.
            timeout (float, optional): HTTP request timeout in seconds.
                Defaults to DEFAULT_TIMEOUT.
            max_retries (int, optional): Maximum number of retry attempts for failed requests.
                Defaults to DEFAULT_MAX_RETRIES.
            retry_delay (float, optional): Base delay between retries in seconds.
                Defaults to DEFAULT_RETRY_DELAY.
            max_connections (int, optional): Maximum number of connections in the connection pool.
                Defaults to DEFAULT_MAX_CONNECTIONS.
            **kwargs (Any): Additional arguments passed to ServerArgs
        """
        super().__init__(router_ip, router_port, timeout, max_retries, retry_delay, **kwargs)
        # Similar to AsyncEngine, track if we need to reload weights
        self._need_reload: bool = True
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
        self.max_connections: int = max_connections

    @asynccontextmanager
    async def _get_session(self) -> aiohttp.ClientSession:
        """Context manager for safe session access with proper connection pooling.

        Yields:
            aiohttp.ClientSession: Session instance for making HTTP requests

        Note:
            This method ensures thread-safe access to the session and handles
            session creation/recreation as needed. If an error occurs during
            session usage, the session will be closed and recreated on next access.
        """
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_connections // 4,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                )
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            try:
                yield self._session
            except Exception:
                # If there's an error, close the session to force recreation
                if self._session and not self._session.closed:
                    await self._session.close()
                    self._session = None
                raise

    async def _make_async_request(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        only_master: bool = True,
    ) -> Dict[str, Any]:
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
            - For non-master nodes (node_rank != 0), returns empty dict immediately
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return await response.json()
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return await response.json()

            except asyncio.TimeoutError:
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            except aiohttp.ClientConnectorError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_retries:
                    raise

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_retries + 1} attempts")

    async def release_memory_occupation(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Release GPU memory occupation temporarily (async version).

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return await self._make_async_request("release_memory_occupation", {"tags": tags})

    async def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Resume GPU memory occupation (async version).

        Similar to AsyncEngine, this method handles first-time weight reloading
        by calling release_memory_occupation if needed.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        # Similar to AsyncEngine, handle first-time reload
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        return await self._make_async_request("resume_memory_occupation", {"tags": tags})

    async def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> Dict[str, Any]:
        """Update model weights from tensor data asynchronously.

        Args:
            serialized_named_tensors (List[str]): List of serialized tensor data
            load_format (Optional[str], optional): Format specification for loading weights.
                Defaults to None.
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to True.

        Returns:
            Dict[str, Any]: Server response containing update status
        """
        return await self._make_async_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    async def flush_cache(self) -> Dict[str, Any]:
        """Flush the cache of the server asynchronously.

        Similar to the sync version, this method retries until the cache
        is successfully flushed. It uses async sleep between retries.

        Returns:
            Dict[str, Any]: Server response indicating cache flush status.
                For non-master nodes, returns empty dict.

        Note:
            Uses retry logic with limited attempts (max_retries * 2) to avoid infinite loops.
            Each retry includes an async delay to allow pending requests to complete.
        """
        if self.node_rank != 0:
            return {}

        # Use retry logic with limited attempts to avoid infinite loops
        for attempt in range(self.max_retries * 2):  # Allow more retries for cache flush
            try:
                async with self._get_session() as session:
                    url = f"http://{self.server_args.host}:{self.server_args.port}/flush_cache"
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
            except Exception as e:
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

            await asyncio.sleep(self.retry_delay)

        logger.error("Failed to flush cache after maximum attempts")
        return {}

    async def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        input_ids: Optional[List[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate text using the SGLang server asynchronously.

        Args:
            prompt (Optional[str], optional): Text prompt for generation. Defaults to None.
            sampling_params (Optional[Dict[str, Any]], optional): Parameters controlling
                text generation sampling. Defaults to None.
            input_ids (Optional[List[int]], optional): Alternative to prompt, direct token IDs input.
                Defaults to None.
            image_data (Optional[Any], optional): Image data for multimodal generation.
                Defaults to None.
            return_logprob (bool, optional): Whether to return log probabilities.
                Defaults to False.
            logprob_start_len (Optional[int], optional): Starting length for log probability calculation.
                Defaults to None.
            top_logprobs_num (Optional[int], optional): Number of top log probabilities to return.
                Defaults to None.
            token_ids_logprob (Optional[List[int]], optional): Specific token IDs for
                log probability calculation. Defaults to None.
            lora_path (Optional[str], optional): Path to LoRA adapter weights. Defaults to None.
            custom_logit_processor (Optional[Callable], optional): Custom logit processing function.
                Defaults to None.

        Returns:
            Dict[str, Any]: Generated text and associated metadata from the server

        Note:
            Either prompt or input_ids should be provided, but not both.
            The response format depends on the server configuration and parameters.
        """
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return await self._make_async_request("generate", payload, timeout=self.timeout, only_master=False)

    async def init_weights_update_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int, group_name: str, backend: str
    ) -> Dict[str, Any]:
        """Initialize a distributed weights update group asynchronously.

        Args:
            master_address (str): Address of the master node for distributed communication
            master_port (int): Port of the master node
            rank_offset (int): Offset for process ranks in the group
            world_size (int): Total number of processes in the distributed group
            group_name (str): Name identifier for the process group
            backend (str): Backend to use for distributed communication (e.g., 'nccl', 'gloo')

        Returns:
            Dict[str, Any]: Server response indicating group initialization status
        """
        return await self._make_async_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    async def update_weights_from_distributed(
        self,
        names: List[str],
        dtypes: List[Any],
        shapes: List[Tuple[int, ...]],
        group_name: str,
        flush_cache: bool = False,
    ) -> Dict[str, Any]:
        """Update model weights from distributed tensors asynchronously.

        Args:
            names (List[str]): List of tensor names to update
            dtypes (List[Any]): List of data types for each tensor (typically torch.dtype)
            shapes (List[Tuple[int, ...]]): List of tensor shapes
            group_name (str): Name of the distributed process group
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to False.

        Returns:
            Dict[str, Any]: Server response indicating distributed update status
        """
        return await self._make_async_request(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
                "flush_cache": flush_cache,
            },
        )

    async def pause_generation(self) -> Dict[str, Any]:
        """Pause text generation on the server asynchronously.

        Returns:
            Dict[str, Any]: Server response indicating pause status
        """
        return await self._make_async_request("pause_generation", {})

    async def continue_generation(self) -> Dict[str, Any]:
        """Continue text generation on the server asynchronously.

        Returns:
            Dict[str, Any]: Server response indicating continuation status
        """
        return await self._make_async_request("continue_generation", {})

    async def async_generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        input_ids: Optional[List[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Async generate method that mirrors AsyncEngine.async_generate interface.

        This method provides compatibility with AsyncEngine's async_generate method
        by forwarding the call to the generate method. It ensures API consistency
        between direct engine usage and HTTP-based engine usage.

        Args:
            prompt (Optional[str], optional): Text prompt for generation. Defaults to None.
            sampling_params (Optional[Dict[str, Any]], optional): Parameters controlling
                text generation sampling. Defaults to None.
            input_ids (Optional[List[int]], optional): Alternative to prompt, direct token IDs input.
                Defaults to None.
            image_data (Optional[Any], optional): Image data for multimodal generation.
                Defaults to None.
            return_logprob (bool, optional): Whether to return log probabilities.
                Defaults to False.
            logprob_start_len (Optional[int], optional): Starting length for log probability calculation.
                Defaults to None.
            top_logprobs_num (Optional[int], optional): Number of top log probabilities to return.
                Defaults to None.
            token_ids_logprob (Optional[List[int]], optional): Specific token IDs for
                log probability calculation. Defaults to None.
            lora_path (Optional[str], optional): Path to LoRA adapter weights. Defaults to None.
            custom_logit_processor (Optional[Callable], optional): Custom logit processing function.
                Defaults to None.

        Returns:
            Dict[str, Any]: Generated text and associated metadata from the server

        Note:
            This method is provided for API compatibility with AsyncEngine.
            It forwards all calls to the generate method.
        """
        return await self.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
        )

    async def close(self) -> None:
        """Close the aiohttp session and clean up resources.

        This method should be called when the adapter is no longer needed
        to ensure proper cleanup of HTTP connections and resources.

        Note:
            This method is safe to call multiple times. If the session is
            already closed or None, this method will do nothing.
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("HTTP session closed")

    async def __aenter__(self) -> "AsyncHttpServerEngineAdapter":
        """Async context manager support.

        Returns:
            AsyncHttpServerEngineAdapter: Self for use in async context
        """
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Cleanup on context exit.

        Args:
            exc_type (Optional[type]): Exception type if an exception occurred
            exc_val (Optional[Exception]): Exception value if an exception occurred
            exc_tb (Optional[Any]): Exception traceback if an exception occurred
        """
        await self.close()

    def __del__(self) -> None:
        """Cleanup when object is destroyed.

        This provides a fallback cleanup mechanism for the aiohttp session
        in case the close() method wasn't called explicitly. Note that this
        is not ideal for async cleanup but provides a safety net.

        Warning:
            This method attempts async cleanup in a sync context, which may
            not always work reliably. It's recommended to explicitly call
            close() or use the async context manager instead.
        """
        if hasattr(self, "_session") and self._session and not self._session.closed:
            # Note: This is not ideal for async cleanup, but provides a fallback
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass  # Ignore cleanup errors during destruction
