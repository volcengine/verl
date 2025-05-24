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
    messages_dumps: List[Dict[str, Any]]
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools: Optional[list[dict]] = None
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
        values["messages_dumps"] = [msg.model_dump() for msg in values["messages"]]

        # input_ids and attention_mask are set to the prompt without the generation prompt.
        #     1. Used as the base to compute generation_prompt_ids.
        #     2. Ensures consistent behavior: input_ids does not include the generation prompt, so it must be added using get_prompt_ids for each AI turn.
        # prompt_ids and prompt_attention_mask are set to the prompt with the generation prompt.
        #     1. Used to derive generation_prompt_ids by subtracting input_ids.
        #     2. Used in the finalize method to extract response_ids from input_ids.
        if tool_schemas := values.get("tool_schemas"):
            tools = values["tools"] = [tool.model_dump() for tool in tool_schemas]
            tokenization_dict_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True, return_dict=True)
            tokens_without_prompt, values["attention_mask"] = tokenization_dict_without_prompt["input_ids"], tokenization_dict_without_prompt["attention_mask"]

            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=True)
            tokens_with_prompt, values["prompt_attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(tokens_with_prompt) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(tokens_with_prompt)} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")
        elif not values.get("input_ids") or not values.get("attention_mask"):
            raise ValueError("input_ids and attention_mask is required for requests without tools")
        else:
            tokenization_dict_without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, return_dict=True)
            values["attention_mask"], values["prompt_attention_mask"] = tokenization_dict_without_prompt["attention_mask"], values["attention_mask"]
            tokens_without_prompt, tokens_with_prompt = tokenization_dict_without_prompt["input_ids"], values["input_ids"]

        values["input_ids"], values["prompt_ids"] = tokens_without_prompt, tokens_with_prompt
        values["position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["prompt_attention_mask"])).tolist()
        values["loss_mask"], values["prompt_loss_mask"] = [0] * len(values["input_ids"]), [0] * len(values["prompt_ids"])
        values["generation_prompt_ids"] = tokens_with_prompt[len(tokens_without_prompt) :]
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool, full_tokens: bool = False) -> None:
        message_len_delta = (len(new_input_ids) - len(self.input_ids)) if full_tokens else len(new_input_ids)
        self.input_ids = new_input_ids if full_tokens else (self.input_ids + new_input_ids)
        attention_mask = [int(attention_mask)] * message_len_delta
        self.attention_mask += attention_mask
        _delta_position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
        last_position_id = self.position_ids[-1]
        _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
        self.loss_mask += [int(loss_mask)] * message_len_delta
        self.position_ids += _position_ids

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def _append_messages(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
        self.messages_dumps.extend([msg.model_dump() for msg in messages])

    def _tokenize_all_messages(self, tokenizer: PreTrainedTokenizer, delta_input_ids_to_check: Optional[list[int]], add_generation_prompt: bool = False) -> None:
        full_input_ids = tokenizer.apply_chat_template(self.messages_dumps, tools=self.tools, add_generation_prompt=add_generation_prompt, tokenize=True)
        if self.tokenization_mode == "sanity_check" and delta_input_ids_to_check is not None:
            assert full_input_ids == self.input_ids + delta_input_ids_to_check, (
                f"Sanity check failed.\nFull tokenization result:\n{tokenizer.decode(full_input_ids, skip_special_tokens=False)}\nFast tokenization result:\n{tokenizer.decode(self.input_ids + delta_input_ids_to_check, skip_special_tokens=False)}"
            )
        self._update_input_ids(full_input_ids, attention_mask=True, loss_mask=False, full_tokens=True)

    def get_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        if self.tokenization_mode == "fast":
            self._update_input_ids(self.generation_prompt_ids, attention_mask=True, loss_mask=False)
        else:
            self._tokenize_all_messages(tokenizer, self.generation_prompt_ids, add_generation_prompt=True)
        return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        content_ids: Optional[List[int]] = None,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self._append_messages([Message(role="assistant", content=content, tool_calls=tool_calls)])

        if self.tokenization_mode != "full":
            if tool_calls or not content_ids:
                # Handles cases where tool calls are incorrectly embedded, such as: I'll call the tool: <tool_call>{"name": ...}</tool_call>. Does this make sense?
                # The code below restructures the text and tool calls parsed by the SGLang tool parser using the chat template.
                # The outcome depends on the SGLang tool parser; for instance, with Qwen, any text after the first tool call is ignored.
                # TODO: Reconsider this approach for RL scenarios: 1. Try to parse as much valid response as possible; 2. Surface the error to the model for learning.
                content_start_pos = len(tokenizer.apply_chat_template(self.messages_dumps[:-1], tools=self.tools, add_generation_prompt=True, tokenize=False))
                content = tokenizer.apply_chat_template(self.messages_dumps, tools=self.tools, add_generation_prompt=False, tokenize=False)[content_start_pos:]
                content_ids = tokenizer.encode(content, add_special_tokens=False)

            if self.tokenization_mode == "fast":
                self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)
                return

        self._tokenize_all_messages(tokenizer, content_ids)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return

        self._append_messages([Message(role="tool", content=content) for content in contents])
        response_token_ids = None
        if self.tokenization_mode != "full":
            response_start_pos = len(tokenizer.apply_chat_template(self.messages_dumps[: -len(contents)], tools=self.tools, add_generation_prompt=False, tokenize=False))
            response_tokens = tokenizer.apply_chat_template(self.messages_dumps, tools=self.tools, add_generation_prompt=False, tokenize=False)[response_start_pos:]
            response_token_ids = tokenizer.encode(response_tokens, add_special_tokens=False)

            if self.tokenization_mode == "fast":
                self._update_input_ids(response_token_ids, attention_mask=True, loss_mask=False)
                return

        self._tokenize_all_messages(tokenizer, response_token_ids)

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
