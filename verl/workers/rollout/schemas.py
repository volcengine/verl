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
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
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
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}


    tokenization_mode: Literal["fast", "full", "sanity_check"]
    generation_prompt_ids: List[int]

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        tools = [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        tokens_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True)
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=True, return_dict=True)
            values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool, full_tokens: bool = False) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request.
        When full_tokens is True, it replaces the input_ids with new_input_ids and updates the attention_mask, position_ids, and loss_mask accordingly.
        When full_tokens is False, it appends new_input_ids to the input_ids and updates the attention_mask, position_ids, and loss_mask accordingly.
        """
        message_len_delta = (len(new_input_ids) - len(self.input_ids)) if full_tokens else len(new_input_ids)
        self.input_ids = new_input_ids if full_tokens else (self.input_ids + new_input_ids)
        attention_mask = [int(attention_mask)] * message_len_delta
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * message_len_delta
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def _fast_tokenize(self, tokenizer: PreTrainedTokenizer, num_messages: int, add_generation_prompt: bool, delta_tokens: Optional[List[int]] = None) -> list[int]:
        """Fast tokenization tokenize the new messages only and append the tokens to the existing input_ids."""

        # Handles cases where tool calls are incorrectly embedded, such as: I'll call the tool: <tool_call>{"name": ...}</tool_call>. Does this make sense?
        # The code below restructures the text and tool calls parsed by the SGLang tool parser using the chat template.
        # The outcome depends on the SGLang tool parser; for instance, with Qwen, any text after the first tool call is ignored.
        # TODO: Reconsider this approach for RL scenarios: 1. Try to parse as much valid response as possible; 2. Surface the error to the model for learning.
        if num_messages and (not delta_tokens or self.messages[-1].tool_calls):
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            content_start_pos = len(tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages[:-num_messages]], tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False))
            content = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=tools, add_generation_prompt=False, tokenize=False)[content_start_pos:]
            delta_tokens = tokenizer.encode(content, add_special_tokens=False)
        return delta_tokens

    def _full_tokenize(self, tokenizer: PreTrainedTokenizer, add_generation_prompt: bool) -> list[int]:
        """Full tokenization tokenizes the entire message history and returns the full tokenization result."""
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True)

    def _tokenize_messages(self, tokenizer: PreTrainedTokenizer, num_messages: int, loss_mask: bool, add_generation_prompt: bool, delta_tokens: Optional[List[int]] = None) -> None:
        """
        Tokenizes messages and updates `input_ids`, `attention_mask`, `position_ids`, and `loss_mask` based on the selected tokenization mode.

        :param num_messages: (Only used in "fast" mode) Specifies the number of most recent messages to tokenize.
        :param add_generation_prompt: (Only used in "full" mode) Indicates whether to include a generation prompt in the tokenized output.
        :param delta_tokens: (Only used in "fast" mode) Tokens to append to `input_ids`. If None, the method tokenizes the last `num_messages` messages.
        """
        match self.tokenization_mode:
            case "fast":
                # Only when tokenizing assistant messages do we set loss_mask to True and exclude the generation prompt from token ids.
                # Therefore, only when loss_mask==True, we include the generation prompt in the calculation of the start position of new message tokens
                self._update_input_ids(self._fast_tokenize(tokenizer, num_messages, loss_mask, delta_tokens), attention_mask=True, loss_mask=loss_mask)
            case "full":
                self._update_input_ids(self._full_tokenize(tokenizer, add_generation_prompt), attention_mask=True, loss_mask=loss_mask, full_tokens=True)
            case "sanity_check":
                full_tokens = self._full_tokenize(tokenizer, add_generation_prompt)
                delta_tokens = self._fast_tokenize(tokenizer, num_messages, loss_mask, delta_tokens)
                assert full_tokens == self.input_ids + delta_tokens, f"Sanity check failed.\nFull tokenization result:\n{tokenizer.decode(full_tokens, skip_special_tokens=False)}\nFast tokenization result:\n{tokenizer.decode(self.input_ids + delta_tokens, skip_special_tokens=False)}"
                self._update_input_ids(full_tokens, attention_mask=True, loss_mask=loss_mask, full_tokens=True)
            case _:
                raise ValueError(f"Unsupported tokenization mode: {self.tokenization_mode}. Supported modes are 'fast', 'full', and 'sanity_check'.")

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        generation_prompt_ids = [] if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids else self.generation_prompt_ids
        if not generation_prompt_ids:
            return self.input_ids
        self._tokenize_messages(tokenizer, num_messages=0, loss_mask=False, add_generation_prompt=True, delta_tokens=generation_prompt_ids)
        return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        content_ids: Optional[List[int]] = None,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        self._tokenize_messages(tokenizer, num_messages=1, loss_mask=True, add_generation_prompt=False, delta_tokens=content_ids)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return
        self.messages.extend([Message(role="tool", content=content) for content in contents])
        self._tokenize_messages(tokenizer, num_messages=len(contents), loss_mask=False, add_generation_prompt=False)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores

        # In case we failed to generate the assistant message and the generation prompt ids were already added to input_ids, remove them from the end of input_ids
        if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids:
            self.input_ids = self.input_ids[: -len(self.generation_prompt_ids)]
            self.attention_mask = self.attention_mask[: -len(self.generation_prompt_ids)]
            self.position_ids = self.position_ids[: -len(self.generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(self.generation_prompt_ids)]

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
