# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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
import socket
from contextlib import asynccontextmanager
from typing import Annotated

import fastapi
from agents import Agent, ModelSettings, RunConfig, Runner
from fastapi import FastAPI
from sglang.srt.utils import get_ip

from .apis import ChatRequest, ChatResponse, UserContext
from .model_provider import CustomModelProvider
from .tools import calc_gsm8k_reward

logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level="DEBUG")
logger = logging.getLogger(__file__)
logger.setLevel("WARN")

HEADER_TRACE_ID = "trace_id"


def _get_host_ip():
    return get_ip()


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("AgentServer started")
    try:
        yield
    except asyncio.CancelledError:
        pass
    print("AgentServer shutdown")


app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Annotated[ChatRequest, fastapi.Body()]):
    """A demo chat function."""
    context = request.context
    model_provider = CustomModelProvider()
    extra_headers = request.extra_headers or {}
    extra_headers.update({HEADER_TRACE_ID: context.trace_id})
    model_settings = ModelSettings(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        extra_headers=extra_headers,
        extra_body=request.extra_body or {},
    )
    agent = Agent[UserContext](
        name="Assistant",
        instructions=request.system_prompt or "You are a helpful assistant.",
        tools=[calc_gsm8k_reward],
    )
    run_config = RunConfig(model_provider=model_provider, model_settings=model_settings)
    try:
        result = await Runner.run(
            agent, request.prompt, context=context, run_config=run_config, max_turns=context.max_turns
        )
    except Exception as e:
        logger.error(f"Error in AgentServer chat: {e}")
        return ChatResponse(context=request.context, response=f"Error: {e}", error=str(e))
    return ChatResponse(context=result.context_wrapper.context, response=result.final_output)
