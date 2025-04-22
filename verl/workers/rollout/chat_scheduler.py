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
import heapq
from typing import Any, Callable, Dict, List
from uuid import uuid4

import aiohttp
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        tokenizer: PreTrainedTokenizer,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig, rollout config.
            model_path: str, model path.
            tokenizer: PreTrainedTokenizer, tokenizer.
            server_addresses: List[str], server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config
        self.model_name = "/".join(model_path.split("/")[-2:])
        self.tokenizer = tokenizer

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

    async def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, Dict[str, Any]], None],
        callback_additional_info: Dict[str, Any],
        **chat_complete_request,
    ):
        """
        Submit a chat completion request to the server with the least number of requests.

        Args:
            callback: Callable[[ChatCompletion, Dict[str, Any]], None], async callback function to handle the response.

                **CAUTION**: the callback function must be async and non-blocking, if you have any blocking operation,
                please move to seperate thread or process pool to avoid blocking the event loop.

            callback_additional_info: Dict[str, Any], additional info to pass to the callback function.

            **chat_complete_request: dict, request parameters same as OpenAI AsyncCompletions.create.
                OpenAI API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        if "extra_headers" not in chat_complete_request:
            chat_complete_request["extra_headers"] = {}

        extra_headers = chat_complete_request["extra_headers"]
        request_id = extra_headers.get("x-request-id", None)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
                extra_headers["x-request-id"] = request_id

            address = self.request_id_to_address[request_id]
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

            request_id = uuid4().hex
            self.request_id_to_address[request_id] = address
            chat_complete_request["extra_headers"]["x-request-id"] = request_id

        # TODO: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
        completions = await self._chat_completions_openai(address, **chat_complete_request)

        await callback(completions, callback_additional_info)

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(
            base_url=f"http://{address}/v1",
            api_key="token-abc123",
        )
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            session = aiohttp.ClientSession()
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123"},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        raise NotImplementedError
