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
import aiohttp
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion


async def chat_completions_aiohttp(address: str, **chat_complete_request) -> ChatCompletion:
    try:
        extra_body = chat_complete_request.pop("extra_body", {})
        chat_complete_request.update(extra_body or {})
        extra_headers = chat_complete_request.pop("extra_headers")
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=f"http://{address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


async def abort_chat_completions_aiohttp(address: str, request_id) -> ChatCompletion:
    pass


async def chat_completions_openai(address: str, api_key: str = "token-abc123", **chat_complete_request) -> ChatCompletion:
    client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key=api_key, timeout=None, max_retries=0)
    return await client.chat.completions.create(**chat_complete_request)
