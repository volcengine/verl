# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    messages_dumps: List[Dict[str, Any]]
    tools: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_response_len: int = 8192
    max_model_len: int = 32768

    generation_prompt_ids: List[int]

    @model_validator(mode="before")
    @classmethod
    def populate_chat_template_info(cls, values):
        messages = values.get("messages")
        if messages is None:
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        values["messages"] = [Message.model_validate(msg) for msg in messages]
        values["messages_dumps"] = [msg.model_dump() for msg in values["messages"]]

        tokenizer = values.pop("tokenizer", None)
        if tokenizer is None:
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        tokens_without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        tokens_with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        values["generation_prompt_ids"] = tokens_with_prompt[len(tokens_without_prompt) :]
        return values

    def append_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        _delta_position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
        last_position_id = self.position_ids[-1]
        _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += _position_ids

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def append_messages(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
        self.messages_dumps.extend([msg.model_dump() for msg in messages])

    def get_prompt_ids(self) -> List[int]:
        self.append_input_ids(self.generation_prompt_ids, attention_mask=True, loss_mask=False)
        return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        content_ids: Optional[List[int]] = None,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        already_over_long: bool = False,
    ) -> None:
        self.append_messages([Message(role="assistant", content=content, tool_calls=tool_calls)])
        if tool_calls is None and content_ids is not None:
            self.append_input_ids(content_ids, attention_mask=True, loss_mask=True)
            return

        # Handles cases where tool calls are incorrectly embedded, such as: I'll call the tool: <tool_call>{"name": ...}</tool_call>. Does this make sense?
        # The code below restructures the text and tool calls parsed by the SGLang tool parser using the chat template.
        # The outcome depends on the SGLang tool parser; for instance, with Qwen, any text after the first tool call is ignored.
        # TODO: Reconsider this approach for RL scenarios: 1. Try to parse as much valid response as possible; 2. Surface the error to the model for learning.
        content_start_pos = len(tokenizer.apply_chat_template(self.messages_dumps[:-1], add_generation_prompt=True, tokenize=False))
        content = tokenizer.apply_chat_template(self.messages_dumps, add_generation_prompt=False, tokenize=False)[content_start_pos:]

        if already_over_long:
            content = content[: -len(tokenizer.eos_token)]
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        self.append_input_ids(content_token_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return

        self.append_messages([Message(role="tool", content=content) for content in contents])
        response_start_pos = len(tokenizer.apply_chat_template(self.messages_dumps[: -len(contents)], add_generation_prompt=False, tokenize=False))
        response_tokens = tokenizer.apply_chat_template(self.messages_dumps, add_generation_prompt=False, tokenize=False)[response_start_pos:]
        response_token_ids = tokenizer.encode(response_tokens, add_special_tokens=False)
        self.append_input_ids(response_token_ids, attention_mask=True, loss_mask=False)

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
