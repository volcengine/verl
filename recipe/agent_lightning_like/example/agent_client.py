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

from typing import Any

import aiohttp

from recipe.agent_lightning_like.agent_client_base import AgentClientBase

from .apis import ChatRequest, UserContext


def extract_prompt_from_dict(data: dict[str, Any]) -> tuple[str, str]:
    """Extract system_prompt and prompt from a sample from RLHFDataset."""

    def _extract_one_message(message: dict[str, Any]) -> tuple[str, str]:
        assert "role" in message and "content" in message
        role = message["role"]
        content = message["content"]
        assert role in ["system", "user"], f"role must be 'system' or 'user', got {role}"
        return role, content

    assert "raw_prompt" in data, "raw_prompt is required, expect return_raw_chat=True in data config"
    messages = data["raw_prompt"]
    assert isinstance(messages, list) and len(messages) >= 1
    role, content = _extract_one_message(messages[0])
    if role == "user":
        return "", content
    system_prompt = content
    assert len(messages) >= 2
    role, content = _extract_one_message(messages[1])
    assert role == "user"
    return system_prompt, content


class AgentClient(AgentClientBase):
    """Demo AgentClient."""

    async def chat(self, trace_id: str, sampling_params: dict[str, Any], **kwargs) -> Any:
        """A demo chat function."""
        assert trace_id, "trace_id is required"

        # kwargs include "max_turns" and non-tensor fields of a data sample from RLHFDataset
        system_prompt, prompt = extract_prompt_from_dict(kwargs)
        # demo Gsm8kTool needs ground_truth
        assert "reward_model" in kwargs and "ground_truth" in kwargs["reward_model"]
        ground_truth = kwargs["reward_model"]["ground_truth"]

        _params = sampling_params.copy()
        max_tokens = _params.pop("max_tokens", 1024)
        temperature = _params.pop("temperature", 0.6)
        top_p = _params.pop("top_p", 0.8)
        # other sampling params passed in extra_body

        context = UserContext(
            trace_id=trace_id,
            ground_truth=ground_truth,
        )
        if kwargs.get("max_turns") is not None:
            context.max_turns = kwargs["max_turns"]

        request = ChatRequest(
            context=context,
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=_params,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_address_full}/chat",
                json=request.model_dump(),
            ) as resp:
                resp.raise_for_status()
                resp_json = await resp.json()
                return resp_json
