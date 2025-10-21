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
"""
Ref: https://python.langchain.com/docs/how_to/custom_chat_model/
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AsyncLLMServerManager
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MaxTokenExceededError(Exception):
    """Indicate that history chat messages + tool message exceeds LLM max_tokens."""

    pass


class ChatModel(BaseChatModel):
    model_name: str = Field(alias="model")
    """The name of the model"""

    client: AsyncLLMServerManager
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_tokens: int
    """Max tokens to generate"""

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

    def bind_tools(self, tools, **kwargs) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.

        Returns:
            A Runnable that returns a message.
        """
        formatted_tools: list = [convert_to_openai_tool(tool) for tool in tools]

        # used to remove system prompt prefix when encoding tool response
        system_prompt = self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        kwargs["system_prompt"] = system_prompt

        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseChatModel]:
        """Ref: https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/"""
        raise NotImplementedError

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate chat completion message.

        Args:
            messages (list[BaseMessage]): List of list of messages.
            stop (Optional[list[str]], optional): Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings. Defaults to None.

        Returns:
            ChatResult: Chat result.
        """
        request_id, prompt_ids, response_mask = await self._preprocess(messages, **kwargs)

        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])

        output = await self.client.generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
        )

        message = await self._postprocess(request_id, prompt_ids, response_mask, output.token_ids, **kwargs)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name

    async def _preprocess(self, messages: list[BaseMessage], **kwargs: Any) -> tuple[str, list[int], list[int]]:
        """Preprocess messages for chat completion.

        To ensure strong consistency with policy model, AsyncLLM server generate response with token in token out
        instead of messages list.

        But all agent frameworks use messages list to represent chat history. To mitigate the gap, we store trajectory
        (prompt_ids, response_mask) in lastest AIMessage.response_metadata.

        1. Encode ToolMessage to token ids.
        2. Retrieve trajectory (prompt_ids, response_mask) from lastest AIMessage.response_metadata.
        3. Append ToolMessage token ids to prompt_ids, and append 0 to response_mask.

        Ref: https://python.langchain.com/docs/concepts/chat_history/

        Args:
            messages (list[BaseMessage]): List of messages.

        Returns:
            tuple[str, list[int], list[int]]: Request id, prompt ids, response mask.
        """
        # messages: [system], human, ai, human|tool, ai, human|tool, ...
        assert messages[-1].type in ["human", "tool"], (
            f"Last message must be human or tool, but got {messages[-1].type}"
        )
        loop = asyncio.get_running_loop()

        # Case 1: initial chat completion: [system], human
        if messages[-1].type == "human" and (len(messages) == 1 or messages[-2].type != "ai"):
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    convert_to_openai_messages(messages),
                    tools=kwargs.get("tools"),
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
            return str(uuid.uuid4()), prompt_ids, []

        # Case 2: follow up chat completion with tool/human response: [system], human, ai, human|tool, ...
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == "ai":
                break
        assert "prompt_ids" in messages[i].response_metadata, "Last message must have prompt_ids in response_metadata"
        assert "response_mask" in messages[i].response_metadata, (
            "Last message must have response_mask in response_metadata"
        )

        # encode tool response
        tool_responses = convert_to_openai_messages(messages[i + 1 :])
        if self.tool_parser == "hermes":
            tool_response_ids = await loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(kwargs["system_prompt"]) :]
        elif self.tool_parser == "gpt-oss":
            # Format tool responses manually
            # since gpt-oss chat template requires tool call messages to parse tool response messages
            # we need to format the tool response messages manually
            tool_response_texts = []
            for tool_msg in tool_responses:
                if tool_msg["role"] == "tool":
                    # Use tool message's name if available (for multiple tool calls)
                    actual_tool_name = tool_msg.get("name", "unknown")
                    if actual_tool_name == "unknown":
                        logger.error(f"actual_tool_name: {actual_tool_name}")
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

            # Tokenize the manually formatted tool responses
            tool_response_text = "".join(tool_response_texts)
            # need to add generation tokens for gpt-oss manually since add_generation_prompt is True
            tool_response_text = add_generation_prompt_for_gpt_oss(tool_response_text)
            logger.debug(f"tool_response_text: {tool_response_text}")

            tool_response_ids = await loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            raise ValueError(f"Unsupported tool parser: {self.tool_parser}")

        # stop generation if response length exceeds max response length
        if len(messages[i].response_metadata["response_mask"]) + len(tool_response_ids) >= self.max_tokens:
            raise MaxTokenExceededError(f"Max response length {self.max_tokens} exceeded")

        # append tool response to prompt
        request_id = messages[i].response_metadata.pop("request_id")
        prompt_ids = messages[i].response_metadata.pop("prompt_ids")
        response_mask = messages[i].response_metadata.pop("response_mask")
        prompt_ids += tool_response_ids
        response_mask += [0] * len(tool_response_ids)

        return request_id, prompt_ids, response_mask

    async def _postprocess(
        self, request_id: str, prompt_ids: list[int], response_mask: list[int], response_ids: list[int], **kwargs: Any
    ) -> AIMessage:
        """Postprocess response_ids when chat completion is done.

        1. Decode response_ids, parse tool calls to AIMessage.
        2. Append response_ids to prompt_ids, and append 1 to response_mask.
        3. Store trajectory (prompt_ids, response_mask) in AIMessage.response_metadata.

        Args:
            request_id (str): Unique request id.
            prompt_ids (list[int]): Input prompt token ids in this chat completion.
            response_mask (list[int]): Response mask before this chat completion.
            response_ids (list[int]): LLM generated token ids in this chat completion.

        Returns:
            AIMessage: Postprocessed message.
        """
        prompt_ids += response_ids
        response_mask += [1] * len(response_ids)

        tool_parser = ToolParser.get_tool_parser(self.tool_parser, self.tokenizer)
        content, function_calls = await tool_parser.extract_tool_calls(response_ids)

        tool_calls, invalid_tool_calls = [], []

        for function_call in function_calls:
            error = None
            try:
                args = json.loads(function_call.arguments)
                if not isinstance(args, dict):
                    error = f"Tool arguments must be a JSON object, got {type(args).__name__}"
            except json.JSONDecodeError as e:
                error = f"Invalid JSON tool arguments: {e}"

            if error:
                logger.warning(error)
                invalid_tool_calls.append(
                    InvalidToolCall(
                        name=function_call.name,
                        args=function_call.arguments,
                        id=str(uuid.uuid4()),
                        error=error,
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(
                        name=function_call.name,
                        args=args,
                        id=str(uuid.uuid4()),
                    )
                )

        message = AIMessage(
            content=content,
            tool_calls=tool_calls[: self.max_parallel_calls],
            invalid_tool_calls=invalid_tool_calls[: self.max_parallel_calls],
            response_metadata={
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "response_mask": response_mask,
            },
        )
        return message


class TruncateStructuredTool(StructuredTool):
    """Structured tool with response truncation."""

    tool_response_truncate_side: str
    """truncate side of tool response: left, middle, right"""

    max_tool_response_length: int
    """max length of tool response"""

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Any:
        tool_response = await super()._arun(*args, config=config, **kwargs)
        tool_response = str(tool_response)

        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

        return tool_response


def convert_to_agent_output(messages: list[BaseMessage], response_length: int) -> AgentLoopOutput:
    """Convert messages to AgentLoopOutput.

    Args:
        messages (List[BaseMessage]): List of messages, last message must be assistant
            with response_metadata containing `prompt_ids` and `response_mask`.
        response_length (int): Max length of response.

    Returns:
        AgentLoopOutput: agent loop output trajectory used for training.
    """
    # skip last tool calls
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type != "tool":
            break
    last_message = messages[i]
    assert last_message.type == "ai", f"Last message must be assistant, but got {last_message.type}"
    assert "prompt_ids" in last_message.response_metadata, "Last message must have prompt_ids in response_metadata"
    assert "response_mask" in last_message.response_metadata, (
        "Last message must have response_mask in response_metadata"
    )

    num_turns = 0
    for i in range(len(messages)):
        if messages[i].type == "system":
            continue
        # parallel tool calls are in single turn
        if i == 0 or messages[i].type != messages[i - 1].type:
            num_turns += 1

    prompt_ids = last_message.response_metadata["prompt_ids"]
    response_mask = last_message.response_metadata["response_mask"]

    response_ids = prompt_ids[-len(response_mask) :]
    prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:response_length],
        response_mask=response_mask[:response_length],
        num_turns=num_turns,
        metrics={},
    )
    return output
