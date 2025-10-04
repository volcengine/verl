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

import json
import logging
import multiprocessing
import os
import time
from typing import Any, Optional

import aiohttp
import ray
import requests
import torch
from sglang_router.launch_server import RouterArgs, launch_router

from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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

    async def _make_async_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        method: str = "POST",
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
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=url,
                json=payload,
            ) as response:
                output = await response.text()
                try:
                    output = json.loads(output)
                    return output
                except Exception as e:
                    print(f"Error: {e}. Output: {output}")
                    return {}
        finally:
            await session.close()

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
        responses = await self._make_async_request("generate", payload)
        return responses
