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
import copy
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import fastapi
import ray
import uvicorn
from omegaconf import DictConfig
from openai.types import Model
from starlette.requests import Request
from starlette.responses import JSONResponse

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

from .message_model import InternalChatCompletion, InternalChoice
from .parser import parse_model_response
from .trajectory import Trajectory, TrajectoryItem

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_MODEL_NAME = "Default"
ERR_PROMPT_TOO_LONG = "conversation too long"


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@ray.remote(num_cpus=4)
class LLMRouter:
    """A OpenAI-compatible server routing requests to LLM servers, and recording trajectories for training."""

    def __init__(self, config: DictConfig, tokenizer, server_handles: list[ray.actor.ActorHandle]):
        self.address = ray.util.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

        self._validate_config(config)
        self.config = config
        self.tokenizer = tokenizer
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.max_model_len = self.prompt_length + self.response_length
        self.created_at = None

        self.model_name = config.lightning_trainer.model_name
        self.tool_call_parser = config.lightning_trainer.tool_call_parser
        self.header_trace_id = config.lightning_trainer.request_header_trace_id
        self._init_traj_store()

    def _validate_config(self, config: DictConfig):
        assert config.get("lightning_trainer") is not None, "config.lightning_trainer is required"
        assert config.lightning_trainer.model_name and isinstance(config.lightning_trainer.model_name, str)
        assert config.lightning_trainer.tool_call_parser and isinstance(config.lightning_trainer.tool_call_parser, str)
        assert config.lightning_trainer.request_header_trace_id and isinstance(
            config.lightning_trainer.request_header_trace_id, str
        )

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print(f"LLMRouter listening on {self.address}:{self.port}")
            self.created_at = int(time.time())
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process and let the trainer handle this.
            print("LLMRouter shutdown, maybe address already in use.")
            raise SystemExit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get LLMRouter server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    def _init_traj_store(self):
        # key: (model_name, trace_id)
        self.traj_store: dict[tuple[str, str], Trajectory] = {}

    def _get_last_traj_item(self, model_name: str, trace_id: str) -> TrajectoryItem | None:
        key = (model_name, trace_id)
        if key not in self.traj_store:
            return None
        return self.traj_store[key].get_last_item()

    def _add_traj_item(self, model_name: str, trace_id: str, item: TrajectoryItem):
        key = (model_name, trace_id)
        if key not in self.traj_store:
            self.traj_store[key] = Trajectory(trace_id=trace_id, model_name=model_name)
        # suppose the llm requests from one session are not overlapped
        # no need to lock
        self.traj_store[key].add_item(item)

    def retrieve_trajectory(self, model_name: str, trace_id: str, clear=True) -> Trajectory | None:
        key = (model_name, trace_id)
        if clear:
            return self.traj_store.pop(key, None)
        else:
            return self.traj_store.get(key, None)

    async def list_models(self) -> JSONResponse:
        model = Model(
            id=self.model_name,
            object="model",
            created=self.created_at,
            owned_by=self.config.actor_rollout_ref.rollout.name,
            root=self.config.actor_rollout_ref.model.path,
            max_model_len=self.max_model_len,
        )
        return JSONResponse(status_code=200, content={"data": [model.model_dump()], "object": "list"})

    async def chat_completion(self, raw_request: Request) -> JSONResponse:
        """Handle chat completion request, and record the trajectory.
        Convert the request as token-in-token-out and forward to llm server manager.
        """
        headers = raw_request.headers
        assert headers and self.header_trace_id in headers, f"{self.header_trace_id} is required in request header"
        trace_id = headers[self.header_trace_id]

        request = await raw_request.json()
        logger.debug(f"LLMRouter receive request {trace_id=}: {request}")
        req = _openai_req_to_internal(request)
        messages = req.messages

        last_traj_item = self._get_last_traj_item(req.model_name, trace_id)
        prev_prompt_ids = last_traj_item.prompt_ids if last_traj_item else []
        prev_response_ids = last_traj_item.response_ids if last_traj_item else []
        prev_response_mask = last_traj_item.response_mask if last_traj_item else []

        loop = asyncio.get_event_loop()
        new_messages, new_token_ids = await loop.run_in_executor(
            None,
            lambda: _get_new_messages_and_tokens(
                self.tokenizer,
                last_traj_item,
                messages,
                tools=req.tools,
            ),
        )

        _prompt_ids_turn = prev_prompt_ids + prev_response_ids + new_token_ids
        logger.debug(f"{trace_id=}, prompt (len={len(_prompt_ids_turn)}): {self.tokenizer.decode(_prompt_ids_turn)}")
        if len(_prompt_ids_turn) >= self.prompt_length + self.response_length:
            logger.error(f"{trace_id=}, prompt too long, len={len(_prompt_ids_turn)}")
            ret = {
                "error": {
                    "type": "error",
                    "code": "ERR_BAD_REQUEST",
                    "message": ERR_PROMPT_TOO_LONG,
                    "param": None,
                }
            }
            # 400 Bad Request
            return JSONResponse(status_code=400, content=ret)

        output = await self.server_manager.generate(
            request_id=trace_id,
            prompt_ids=_prompt_ids_turn,
            sampling_params=req.sampling_params,
        )
        new_response_ids = output.token_ids
        logger.debug(f"{trace_id=}, response (len={new_response_ids}): {self.tokenizer.decode(new_response_ids)}")

        resp_message = await parse_model_response(
            new_response_ids,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=None,
        )
        logger.debug(f"{trace_id=}, parsed message: {resp_message}")

        message_traj = resp_message.model_dump()
        messages.append(message_traj)
        new_messages.append(message_traj)

        prompt_ids = prev_prompt_ids if prev_prompt_ids else new_token_ids
        if prev_response_ids:
            response_ids = prev_response_ids + new_token_ids + new_response_ids
            response_mask = prev_response_mask + [0] * len(new_token_ids) + [1] * len(new_response_ids)
        else:
            response_ids = new_response_ids
            response_mask = [1] * len(new_response_ids)

        traj_item = TrajectoryItem(
            messages=messages,
            new_messages=new_messages,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            extra_info={},
        )
        self._add_traj_item(req.model_name, trace_id, traj_item)
        logger.debug(f"{trace_id=} add trajectory item: {traj_item.model_dump()}")

        if resp_message.tool_calls:
            finish_reason = "tool_calls"
        elif len(response_ids) >= self.response_length:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        choice = InternalChoice(
            index=0,
            message=resp_message,
            finish_reason=finish_reason,
        )
        ret = InternalChatCompletion(
            id=f"chatcmpl-{uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model_name,
            choices=[choice],
        )
        logger.debug(f"LLMRouter send response {trace_id=}: {ret.model_dump()}")
        return JSONResponse(status_code=200, content=ret.model_dump())

    async def completion(self, raw_request: Request) -> JSONResponse:
        raise NotImplementedError


@dataclass
class CompletionReqInternal:
    stream: bool
    model_name: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    sampling_params: dict[str, Any]
    extra_args: dict[str, Any]


def _openai_req_to_internal(req: dict[str, Any]) -> CompletionReqInternal:
    """Convert OpenAI request to dict."""
    req = copy.deepcopy(req)

    assert "messages" in req
    messages = req.pop("messages")
    assert isinstance(messages, list) and len(messages) >= 1
    model_name = req.pop("model", DEFAULT_MODEL_NAME)
    tools = req.pop("tools", None)
    stream = req.pop("stream", False)
    assert stream is False, "streaming response is not supported yet"
    logprobs = req.pop("logprobs", False)
    if logprobs:
        logger.warning("logprobs is not supported yet, ignore it")
    # max_tokens & n controlled by rollout engine
    req.pop("max_tokens", None)
    n = req.pop("n", 1)
    assert n == 1, "only support n=1"

    sampling_params = {}
    sampling_param_keys = [
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ]
    for key in sampling_param_keys:
        if key in req:
            sampling_params[key] = req.pop(key)

    ## TODO: configurable ignore keys
    ignore_keys = ["extra_headers", "extra_query", "extra_body", "timeout"]
    for key in list(req.keys()):
        if key in ignore_keys:
            req.pop(key)
        else:
            logger.warning(f"{key} is not supported in LLMRouter yet, ignore it")

    return CompletionReqInternal(
        stream=stream,
        model_name=model_name,
        messages=messages,
        tools=tools,
        sampling_params=sampling_params,
        extra_args=req,
    )


def _get_new_messages_and_tokens(
    tokenizer,
    last_item: TrajectoryItem | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[dict], list[int]]:
    """Get new messages and prompt tokens compared to last trajectory item."""
    messages = copy.deepcopy(messages)
    if last_item is None:
        new_tokens = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)
        return messages, new_tokens
    last_messages = last_item.messages
    assert len(messages) > len(last_messages), "new messages should be more than last messages"

    new_messages = messages[len(last_messages) :]
    # suppose tools do not change in one session
    new_tokens = _encode_new_messages(tokenizer, new_messages)
    ## TODO: check prompt ids consistency
    return new_messages, new_tokens


def _encode_new_messages(tokenizer, new_messages: list[dict]) -> list[int]:
    """Encode new messages to token ids."""
    assert len(new_messages) > 0, "new_messages should not be empty"
    assert new_messages[0]["role"] != "assistant", "new message should not start with assistant"

    BASE_CHAT_HISTORY = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
        {"role": "assistant", "content": "I am an assistant."},
    ]
    base_token_ids = tokenizer.apply_chat_template(BASE_CHAT_HISTORY, add_generation_prompt=False)
    messages = BASE_CHAT_HISTORY + new_messages
    token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    assert len(token_ids) > len(base_token_ids), "token_ids should be more than base_token_ids"
    new_token_ids = token_ids[len(base_token_ids) :]
    return new_token_ids
