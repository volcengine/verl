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
import socket
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Optional

import fastapi
import ray
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__file__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class AsyncServerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self):
        self.address = ray.util.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print(f"FastAPI listen on {self.address}:{self.port}")
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> tuple[str, int]:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    @abstractmethod
    async def chat_completion(self, raw_request: Request) -> JSONResponse:
        """OpenAI chat completion API.

        Args:
            raw_request (Request): raw json request

        Returns:
            JSONResponse: json response

        API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        """Generate response ids given prompt ids.

        Args:
            prompt_ids (List[int]): prompt ids
            sampling_params (Dict[str, Any]): sampling params
            request_id (str): request id

        Returns:
            List[int]: response ids
        """
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        """Init async LLM engine."""
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        """Wake up engine to load model weights and build kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep engine to offload model weights and discard kv cache."""
        raise NotImplementedError


def async_server_class(
    rollout_backend: str, rollout_backend_module: Optional[str] = None, rollout_backend_class: Optional[str] = None
) -> type[AsyncServerBase]:
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend type (alias), should be "vllm" or "sglang".
        rollout_backend_module: Optional[str], import path of the rollout backend.
        rollout_backend_class: Optional[str], class name of the rollout backend.

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    if rollout_backend_class is None and rollout_backend_module is None:
        # If both are None, use the default backend class
        # Do not change the original import behavior
        # importlib.import_module and from ... import ... have subtle differences in ray

        if rollout_backend == "vllm":
            from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

            return AsyncvLLMServer
        elif rollout_backend == "sglang":
            from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSGLangServer

            return AsyncSGLangServer
        else:
            raise NotImplementedError(f"rollout backend {rollout_backend} is not supported")

    if rollout_backend_module is None or rollout_backend_class is None:
        raise ValueError("rollout_backend_module and rollout_backend_class must be both provided for customization")

    from verl.utils.import_utils import load_extern_type

    return load_extern_type(rollout_backend_module, rollout_backend_class)
