# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
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

from copy import deepcopy

import ray
from agentlightning.instrumentation.vllm import instrument_vllm
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote(num_cpus=1)
class TokenResponseVllmServer(_unwrap_ray_remote(AsyncvLLMServer)):
    """
    Custom vLLM server for AgentLightning that extends AsyncvLLMServer with instrumentation.

    Inheritance: AsyncvLLMServer -> AsyncServerBase
    - AsyncvLLMServer: Base class that wraps vLLM AsyncLLM with external Ray distributed executor
    - Handles OpenAI-compatible API endpoints and vLLM engine initialization

    Customizations:
    1. Adds vLLM instrumentation via instrument_vllm() for monitoring/tracing
    2. Disables multi-turn tool configuration by setting tool_config_path to "/dev/null"
    3. Ray remote decorator for distributed execution
    """

    def __init__(self, *args, **kwargs):
        # Add instrumentation for monitoring and tracing vLLM operations
        instrument_vllm()
        super().__init__(*args, **kwargs)

        # Customize configuration to disable multi-turn tool usage
        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        # The response here is non-standard, so we need to handle it differently.
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return JSONResponse(content=generator.model_dump())
