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
from pydantic import BaseModel, Field, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "I am a user."}]


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
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None

    metadata: Dict[str, Any] = Field(default_factory=dict, exclude=True)


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
    langgraph_input_kwargs: Dict[str, Any] = {}
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
    chat_template_kwargs: Dict[str, Any] = {}

    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool

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

        tools = [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas")) else None
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True, return_dict=True, **values.get("chat_template_kwargs", {}))
            values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer, chat_template_kwargs: Optional[dict[str, Any]] = None) -> list[int]:
        chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else self.chat_template_kwargs
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        base_end_pos = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False, **chat_template_kwargs))
        generation_prompt = tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)[base_end_pos:]
        generation_prompt_ids = tokenizer.encode(generation_prompt, add_special_tokens=False)
        if self.input_ids[-len(generation_prompt_ids) :] != generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=tools, add_generation_prompt=True, tokenize=True, **chat_template_kwargs)
        else:
            return self.input_ids

    def add_messages(self, tokenizer: PreTrainedTokenizer, messages: list[Message], loss_mask: bool = False, exclude_generation_prompt: bool = False, chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        self.messages.extend(messages)
        chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else self.chat_template_kwargs
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, *messages],
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            **chat_template_kwargs,
        )
        base_end_pos = len(
            tokenizer.apply_chat_template(
                BASE_CHAT_HISTORY,
                tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None),
                add_generation_prompt=exclude_generation_prompt,
                tokenize=False,
                **chat_template_kwargs,
            )
        )
        content_ids = tokenizer.encode(content[base_end_pos:], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=loss_mask)

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.add_messages(tokenizer, [Message(role="assistant", content=content, tool_calls=tool_calls)], loss_mask=True, exclude_generation_prompt=True, chat_template_kwargs=chat_template_kwargs)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str], chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        if contents:
            self.add_messages(tokenizer, [Message(role="tool", content=content) for content in contents], chat_template_kwargs=chat_template_kwargs)

    def add_user_message(self, tokenizer: PreTrainedTokenizer, content: str, chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        self.add_messages(tokenizer, [Message(role="user", content=content)], chat_template_kwargs=chat_template_kwargs)

    def add_system_message(self, tokenizer: PreTrainedTokenizer, content: str, chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        self.add_messages(tokenizer, [Message(role="system", content=content)], chat_template_kwargs=chat_template_kwargs)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def add_full_conversation(self, tokenizer: PreTrainedTokenizer, prompt_message_cnt: int, full_conversation: list[dict], tools: list[dict] = None) -> "AsyncRolloutRequest":
        req = AsyncRolloutRequest(
            request_id=self.request_id,
            batch_data_id=self.batch_data_id,
            rollout_offset=self.rollout_offset,
            state=AsyncRolloutRequestStateEnum.COMPLETED,
            messages=full_conversation[:prompt_message_cnt],
            tool_schemas=[OpenAIFunctionToolSchema.model_validate(tool) for tool in tools] if tools else None,
            tools_kwargs=self.tools_kwargs,
            langgraph_input_kwargs=self.langgraph_input_kwargs,
            chat_template_kwargs=self.chat_template_kwargs,
            response_ids=[],
            response_attention_mask=[],
            response_position_ids=[],
            response_loss_mask=[],
            reward_scores={},
            max_prompt_len=self.max_prompt_len,
            max_response_len=self.max_response_len,
            max_model_len=self.max_model_len,
            use_inference_chat_template=self.use_inference_chat_template,
            enable_tokenization_sanity_check=self.enable_tokenization_sanity_check,
            tokenizer=tokenizer,
        )

        messages = [Message.model_validate(msg) for msg in full_conversation[prompt_message_cnt:]]
        tool_messages = []
        for message in messages:
            if message.role == "tool":
                tool_messages.append(message)
                continue

            if tool_messages:
                req.add_tool_response_messages(tokenizer, [tool_message.content for tool_message in tool_messages], tool_messages[0].metadata.get("chat_template_kwargs"))
                tool_messages = []

            match message.role:
                case "assistant":
                    req.get_generation_prompt_ids(tokenizer, chat_template_kwargs=message.metadata.get("chat_template_kwargs"))
                    req.add_assistant_message(tokenizer, message.content, tool_calls=message.tool_calls, chat_template_kwargs=message.metadata.get("chat_template_kwargs"))
                case "user":
                    req.add_user_message(tokenizer, message.content, chat_template_kwargs=message.metadata.get("chat_template_kwargs"))
                case "system":
                    req.add_system_message(tokenizer, message.content, chat_template_kwargs=message.metadata.get("chat_template_kwargs"))
                case _:
                    raise ValueError(f"Unsupported message role: {message.role}")
        return req

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.enable_tokenization_sanity_check:
            full_tokens = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=True, **self.chat_template_kwargs)
            if self.input_ids != full_tokens:
                logger.warning("Inconsistent training and inference tokenization detected. This may lead to unexpected behavior during training. Please review your chat template to determine if this is intentional. For more information, refer to the multiturn README.md.")
                logger.info(f"Inference tokenization result:\n{tokenizer.decode(full_tokens, skip_special_tokens=False)}\ntraining content:\n{tokenizer.decode(self.input_ids, skip_special_tokens=False)}")

        # In case we failed to generate the assistant message and the generation prompt ids were already added to input_ids, remove them from the end of input_ids
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        base_end_pos = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False, **self.chat_template_kwargs))
        generation_prompt = tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs)[base_end_pos:]
        generation_prompt_ids = tokenizer.encode(generation_prompt, add_special_tokens=False)
        if self.input_ids[-len(generation_prompt_ids) :] == generation_prompt_ids:
            self.input_ids = self.input_ids[: -len(generation_prompt_ids)]
            self.attention_mask = self.attention_mask[: -len(generation_prompt_ids)]
            self.position_ids = self.position_ids[: -len(generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(generation_prompt_ids)]

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
