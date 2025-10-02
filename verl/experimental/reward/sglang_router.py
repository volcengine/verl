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
import multiprocessing
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import aiohttp
import ray
import requests
import torch
from sglang_router.launch_server import RouterArgs, launch_router

from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 10000
DEFAULT_MAX_WAIT_TIME = 300.0


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


@ray.remote
class SGLangRouter:
    """Router for SGLang."""

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        worker_urls: list[str],
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_start_wait_time: float = DEFAULT_MAX_WAIT_TIME,
        **kwargs,
    ):
        """
        Initialize the router.

        Args:
            router_ip (str): IP address of the router.
            router_port (int): Port number of the router.
            worker_urls (list[str]): List of worker URLs.
            timeout (float, optional): Timeout for requests in seconds.
            max_attempts (int, optional): Maximum number of retry attempts.
            retry_delay (float, optional): Delay between retry attempts in seconds.
            max_connections (int, optional): Maximum number of concurrent connections.
            max_start_wait_time (float, optional): Maximum time to wait for router startup in seconds.
            **kwargs: Additional keyword arguments.
        """
        self.timeout: float = timeout
        self.max_attempts: int = max_attempts
        self.retry_delay: float = retry_delay
        self.max_connections: int = max_connections
        self.max_start_wait_time: float = max_start_wait_time

        self.router_address = (
            f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
        )
        router_args = RouterArgs(
            host=router_ip,
            port=router_port,
            worker_urls=worker_urls,
            **kwargs,
        )
        self.router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
        self.router_process.start()
        self._wait_for_health_check()

    def _wait_for_health_check(self, max_wait_time=300, timeout=30):
        start_time = time.time()
        url = f"http://{self.router_address}/health"
        with requests.Session() as session:
            while time.time() - start_time < max_wait_time:
                if not self.router_process.is_alive():
                    raise RuntimeError("Router process is not alive.")
                try:
                    response = session.get(url, timeout=timeout)
                    if response.status_code == 200:
                        break
                except requests.RequestException as e:
                    logger.debug(f"Health check failed: {e}")

                time.sleep(2)
            else:
                self.router_process.terminate()
                raise RuntimeError(f"Router health check failed after {max_wait_time} seconds.")

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
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
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
        """

        url = f"http://{self.router_address}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return await _read_async_response(response)
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return await _read_async_response(response)

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

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ):
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "image_data": image_data,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        responses = await self._make_async_request("generate", payload, timeout=self.timeout)
        return responses
