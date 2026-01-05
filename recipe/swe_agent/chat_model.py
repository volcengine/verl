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
import time
import uuid
import copy
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from pydantic import Field

from sweagent.agent.models import AbstractModel
from sweagent.types import History, HistoryItem, Trajectory, TrajectoryStep

from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class MaxTokenExceededError(Exception):
    """Indicate that history chat messages + tool message exceeds LLM max_tokens."""

    pass


class ChatModel(AbstractModel):
    model_name: str = Field(alias="model")
    """The name of the model"""

    client: Any
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_model_len: int
    """Max model context length"""

    tool_parser: str = "hermes"
    """Tool parser for the model"""

    max_parallel_calls: int = 1
    """Max parallel tool calls"""

    temperature: float = 1.0
    """Temperature for sampling"""

    top_p: float = 1.0
    """Top p for sampling"""

    repetition_penalty: float = 1.0
    """Repetition penalty for sampling"""

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

    def query(self, history: list[dict[str, Any]], **kwargs) -> list[dict] | dict:
        messages = self.history_to_messages(history)
        request_id, prompt_ids, response_mask = asyncio.run(self._preprocess(messages, **kwargs))

        if len(prompt_ids) > self.max_model_len:
            print(f"prompt_ids length {len(prompt_ids)} exceeds max_model_len {self.max_model_len}")
            raise MaxTokenExceededError(f"prompt_ids length {len(prompt_ids)} exceeds max_model_len {self.max_model_len}")

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])

        token_output = asyncio.run(self.client.generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
        ))

        response = asyncio.run(self._postprocess(request_id, prompt_ids, response_mask, token_output.token_ids, **kwargs))
        return response

    def history_to_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history = copy.deepcopy(history)

        def get_role(history_item: dict[str, Any]) -> str:
            return history_item["role"]

        messages = []
        for history_item in history:
            role = get_role(history_item)
            if role == "tool":
                message = {
                    "role": role,
                    "content": history_item["content"],
                    # Only one tool call per observations
                    "tool_call_id": history_item["tool_call_ids"][0],  # type: ignore
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                message = {
                    "role": role, 
                    "content": history_item["content"], 
                    "tool_calls": tool_calls,
                    "response_metadata": history_item.get("raw", ""),
                }
                if thinking_blocks := history_item.get("thinking_blocks"):
                    message["thinking_blocks"] = thinking_blocks
            else:
                message = {
                    "role": role, 
                    "content": history_item["content"],
                    "response_metadata": history_item.get("raw", ""),
                }
            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]
            messages.append(message)
        return messages

    @property
    def system_prompt(self):
        if hasattr(self, "_system_prompt"):
            return self._system_prompt

        # used to remove system prompt prefix when encoding tool response
        try:
            self._system_prompt = self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        except Exception:
            # Qwen3-Coder-30B-A3B-Instruct
            self._system_prompt = []
        return self._system_prompt

    async def _preprocess(self, messages: list[dict[str, Any]], **kwargs) -> tuple[str, list[int], list[int]]:
        """Preprocess messages for chat completion.
        Args:
            messages (list[dict[str, str]]): List of messages.

        Returns:
            tuple[str, list[int], list[int]]: Request id, prompt ids, response mask.
        """
        chat_messages = []
        for message in messages:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1, "user message must contain one text"
                chat_messages.append({"role": "user", "content": message["content"][0]["text"]})
            else:
                chat_messages.append(message)
        messages = chat_messages

        # messages: [[system], user, assistant, tool|user, asssistant, tool|user]
        assert messages[-1]["role"] in ["user", "tool"], (
            f"Last message must be user or tool, but got {messages[-1]['role']}"
        )
        loop = asyncio.get_running_loop()

        # Case 1: initial chat completion: [system], user
        if messages[-1]["role"] == "user" and (len(messages) == 1 or messages[-2]["role"] != "assistant"):
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=kwargs.get("tools"),
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
            return str(uuid.uuid4()), prompt_ids, []

        # Case 2: follow up chat completion with tool/user response: [system], user, assistant, tool|user, ...
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant" and messages[i].get("tool_calls") is not None:
                break

        assert "response_metadata" in messages[i], "Last message must have response_metadata"
        response_metadata = messages[i]["response_metadata"]
        assert "prompt_ids" in response_metadata, "Last message must have prompt_ids in response_metadata"
        assert "response_mask" in response_metadata, "Last message must have response_mask in response_metadata"

         # encode tool response
        tool_responses = messages[i + 1 :]
        tool_response_ids = await loop.run_in_executor(
            None,
            lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        tool_response_ids = tool_response_ids[len(self.system_prompt) :]

        # stop generation if exceeds max model length
        if len(response_metadata["prompt_ids"]) + len(tool_response_ids) >= self.max_model_len:
            raise MaxTokenExceededError(f"Max model length {self.max_model_len} exceeded")

        # append tool response to prompt
        request_id = response_metadata.pop("request_id")
        prompt_ids = response_metadata.pop("prompt_ids")
        response_mask = response_metadata.pop("response_mask")
        prompt_ids += tool_response_ids
        response_mask += [0] * len(tool_response_ids)

        return request_id, prompt_ids, response_mask

    async def _postprocess(self, request_id: str, prompt_ids: list[int], response_mask: list[int], response_ids: list[int], **kwargs: Any) -> dict:
        """Postprocess response_ids when chat completion is done.

        Args:
            request_id (str): Unique request id.
            prompt_ids (list[int]): Input prompt token ids in this chat completion.
            response_mask (list[int]): Response mask before this chat completion.
            response_ids (list[int]): LLM generated token ids in this chat completion.

        Returns:
            CompletionResponse: Postprocessed message.
        """
        prompt_ids += response_ids
        response_mask += [1] * len(response_ids)

        tool_parser = ToolParser.get_tool_parser(self.tool_parser, self.tokenizer)
        tools = [OpenAIFunctionToolSchema(**tool) for tool in kwargs.get("tools", [])]
        content, function_calls = await tool_parser.extract_tool_calls(response_ids, tools=tools)

        tool_calls = []
        for function_call in function_calls:
           tool_call = ChatCompletionMessageFunctionToolCall(
               function=function_call.model_dump(), type="function", id=str(uuid.uuid4())
           )
           tool_calls.append(tool_call)
        message = {
            "role": "assistant",
            "message": content,
            "tool_calls": [tool.model_dump() for tool in tool_calls[: self.max_parallel_calls]],
            "raw": {
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "response_mask": response_mask,
            }
        }
        return message
