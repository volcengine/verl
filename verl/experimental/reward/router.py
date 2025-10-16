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

import httpx
import ray
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def launch_router(worker_urls: list[str]):
    router_ip = ray.util.get_node_ip_address().strip("[]")
    router_port, _ = get_free_port(router_ip)
    router_address = (
        f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
    )

    router_process = multiprocessing.Process(
        target=run_router,
        args=(
            router_ip,
            router_port,
            worker_urls,
        ),
    )
    router_process.daemon = True
    router_process.start()
    time.sleep(3)
    assert router_process.is_alive()

    logger.info(f"Router is running on {router_address}")
    return router_process, router_address


def run_router(router_ip: str, router_port: int, worker_urls: list[str]):
    router = NaiveRouter(worker_urls=worker_urls, verbose=False)
    uvicorn.run(router.app, host=router_ip, port=router_port, log_level="warning")


class NaiveRouter:
    def __init__(self, worker_urls: list[str], max_connections: int = 256, verbose: bool = False) -> None:
        """A minimal async load-balancing router."""
        self.verbose = verbose
        self.app = FastAPI()
        self.worker_urls = worker_urls
        self.request_counts = {url: 0 for url in worker_urls}

        self.app = FastAPI()

        # Async HTTP client
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(None),
        )

        # Register a catch-all route to proxy requests
        self.app.api_route("/{path:path}", methods=["GET", "POST"])(self.proxy)

    async def proxy(self, request: Request, path: str):
        """Proxy all requests to a worker URL."""
        if not self.worker_urls:
            return JSONResponse(status_code=503, content={"error": "No available workers"})

        worker_url = self._select_worker()
        target_url = f"{worker_url}/{path}"

        if self.verbose:
            print(f"[router] Forwarding request → {target_url}")

        # Copy request data
        body = await request.body()
        headers = dict(request.headers)

        try:
            # Send request to worker
            response = await self.client.request(request.method, target_url, content=body, headers=headers)

            # Read response
            content = await response.aread()
            content_type = response.headers.get("content-type", "")

            # Try return JSON if possible
            try:
                data = json.loads(content)
                return JSONResponse(content=data, status_code=response.status_code)
            except Exception:
                return Response(
                    content=content,
                    status_code=response.status_code,
                    media_type=content_type or None,
                )
        finally:
            self._release_worker(worker_url)

    def _select_worker(self) -> str:
        """Select the least-loaded worker (simple round-robin by request count)."""
        url = min(self.request_counts, key=self.request_counts.get)
        self.request_counts[url] += 1
        return url

    def _release_worker(self, url: str):
        """Mark worker as free after request completes."""
        self.request_counts[url] = max(0, self.request_counts[url] - 1)
