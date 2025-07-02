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
import difflib
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


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
    content: str | Dict[str, Any] | List[Dict[str, Any]]
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"
    INTERACTING = "interacting"


class TokenizationSanityCheckModeEnum(str, Enum):
    """The enum for tokenization sanity check mode."""

    OFF = "off"
    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    uid: str | None = ""
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    multi_modal_keys: Optional[List[str]] = None
    multi_modal_data: Optional[Dict[str, Any]] = None
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    interaction_kwargs: Dict[str, Any] = {}
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

    use_inference_chat_template: bool
    tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (processing_class := values.pop("processing_class", None)):
            raise ValueError("processing_class is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        # If there is no multi_modal_keys, we assume the multi-modal data is image and video.
        if not values.get("multi_modal_keys"):
            values["multi_modal_keys"] = ["image", "video"]
        if not values.get("multi_modal_data"):
            values["multi_modal_data"] = {key: [] for key in values["multi_modal_keys"]}
        else:
            # check if all multi_modal_keys are in multi_modal_data
            for key in values["multi_modal_keys"]:
                if key not in values["multi_modal_data"]:
                    values["multi_modal_data"][key] = []

        tools = (
            [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        )

        multi_modal_data = values["multi_modal_data"]
        tokens_without_prompt = cls._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = cls._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )

            values["input_ids"], values["attention_mask"] = (
                tokenization_dict_with_prompt["input_ids"],
                tokenization_dict_with_prompt["attention_mask"],
            )
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an
                # error for this case in the future.
                logger.warning(
                    f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len "
                    f"{max_prompt_len} after applied chat template with tools."
                )

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(
            torch.tensor(values["attention_mask"])
        ).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        values["base_conv_wo_gen_prompt_end_pos"] = len(
            cls._handle_apply_chat_template(
                processing_class,
                BASE_CHAT_HISTORY,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
        )
        values["base_conv_with_gen_prompt_end_pos"] = len(
            cls._handle_apply_chat_template(
                processing_class,
                BASE_CHAT_HISTORY,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
        )

        return values

    @staticmethod
    def _handle_apply_chat_template(
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        messages: List[Message],
        multi_modal_data: Dict[str, Any],
        tools: Optional[List[OpenAIFunctionToolSchema]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        return_dict: bool = False,
    ):
        if isinstance(processing_class, PreTrainedTokenizer) or isinstance(processing_class, PreTrainedTokenizerFast):
            if any(len(values) > 0 for values in multi_modal_data.values()):
                logger.warning(
                    "There is multi_modal_data but you are not using a processor. Multi-modal data will be ignored."
                )
            return processing_class.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                return_dict=return_dict,
            )
        elif isinstance(processing_class, ProcessorMixin):
            raw_prompt = processing_class.apply_chat_template(
                messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False
            )
            if not tokenize:
                return raw_prompt

            # When we update multi_model_keys, we also need to update this logic
            images = images if len(images := multi_modal_data.get("image", [])) > 0 else None
            videos = videos if len(videos := multi_modal_data.get("video", [])) > 0 else None
            model_inputs = processing_class(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            assert model_inputs["input_ids"].shape[0] == 1, "input_ids should be a 1D array"
            model_inputs = {k: v[0].tolist() if hasattr(v, "tolist") else v for k, v in model_inputs.items()}
            if return_dict:
                return model_inputs
            else:
                return model_inputs["input_ids"]
        else:
            raise ValueError(f"Unsupported processing class type: {type(processing_class)}")

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (
            compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)
        ).tolist()

        assert (
            len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        ), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def get_generation_prompt_ids(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin]
    ) -> list[int]:
        generation_prompt_ids = (
            []
            if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids
            else self.generation_prompt_ids
        )
        if generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            generation_prompt_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
            return generation_prompt_ids
        else:
            return self.input_ids

    def add_user_message(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        content: str,
    ) -> None:
        self.messages.append(Message(role="user", content=content))
        messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
        # Inference, it is pure text.
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[self.base_conv_wo_gen_prompt_end_pos :]
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def add_assistant_message(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))

        messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
        # Inference, it is pure text.
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[self.base_conv_with_gen_prompt_end_pos :]
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin], contents: list[str]
    ) -> None:
        if not contents:
            return

        self.messages.extend([Message(role="tool", content=content) for content in contents])

        messages = [*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # Currently we don't support tool creates multi-modal data
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[self.base_conv_wo_gen_prompt_end_pos :]
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def _get_prompt_diffs(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        full_prompt_ids: List[int],
        current_prompt_ids: List[int],
        diff_surrounding_chars: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get differences between full prompt and current prompt with surrounding context.

        This function helps debug tokenization mismatches by showing the differences between
        full prompt and current prompt with surrounding context. Instead of just showing
        the exact diff, it includes additional tokens before and after to help locate
        the issue in the chat template.

        For example, if the actual diff is a newline change from "\n\n" to "\n", with
        diff_surrounding_chars the output might look like:

        full_prompt_chunk:    "<|im_start|>assistant\n\nI think..."
        current_prompt_chunk: "<|im_start|>assistant\nI think..."

        This context makes it much easier to identify where in the chat template the
        mismatch occurs.

        Args:
            processing_class: The processing class to use for decoding the token IDs
            full_prompt_ids: Token IDs from applying chat template to all messages at once
            current_prompt_ids: Token IDs from incremental chat template application
            diff_surrounding_chars: Number of surrounding characters to include for context (default: 10)

        Returns:
            List of dicts containing the differing chunks with context and their indices
        """
        full_prompt = processing_class.decode(full_prompt_ids, skip_special_tokens=False)
        current_prompt = processing_class.decode(current_prompt_ids, skip_special_tokens=False)
        s = difflib.SequenceMatcher(None, full_prompt, current_prompt, autojunk=False)
        diffs = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "equal":
                continue

            # Get the surrounding context for better readability
            start_i = max(0, i1 - diff_surrounding_chars)
            end_i = min(len(full_prompt), i2 + diff_surrounding_chars)
            start_j = max(0, j1 - diff_surrounding_chars)
            end_j = min(len(current_prompt), j2 + diff_surrounding_chars)

            diffs.append(
                {
                    "full_prompt_chunk": full_prompt[start_i:end_i],
                    "current_prompt_chunk": current_prompt[start_j:end_j],
                    "indices": (start_i, end_i, start_j, end_j),
                }
            )
        return diffs

    def finalize(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        reward_scores: Dict[str, List[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.tokenization_sanity_check_mode != TokenizationSanityCheckModeEnum.OFF:
            # When there is a diff, we log the diffs with diff_surrounding_chars context
            diff_surrounding_chars = 10

            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            full_prompt_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )

            if diffs := self._get_prompt_diffs(
                processing_class, full_prompt_ids, self.input_ids, diff_surrounding_chars=diff_surrounding_chars
            ):
                log_warning = False
                if self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.STRICT:
                    log_warning = True
                elif self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE:
                    non_strippable_diffs_exist = any(
                        d["full_prompt_chunk"].strip() or d["current_prompt_chunk"].strip() for d in diffs
                    )
                    if non_strippable_diffs_exist:
                        log_warning = True

                if log_warning:
                    mode_str = f" ({self.tokenization_sanity_check_mode.value})"
                    logger.warning(
                        f"Inconsistent training and inference tokenization detected{mode_str}. This may lead to "
                        f"unexpected behavior during training. Please review your chat template to determine if this "
                        f"is intentional. For more information, refer to the multiturn README.md."
                    )
                    logger.warning(
                        f"Showing {diff_surrounding_chars} characters before and after the diffs for context and "
                        f"better readability."
                    )
                    diff_details_list = []
                    for d in diffs:
                        i1, i2, j1, j2 = d["indices"]
                        diff_details_list.append(
                            f"idx {i1}:{i2} -> {j1}:{j2} | full_prompt_chunk: {repr(d['full_prompt_chunk'])} | "
                            f"current_prompt_chunk: {repr(d['current_prompt_chunk'])}"
                        )
                    diff_details = "\n".join(diff_details_list)
                    logger.warning(f"Found differences:\n{diff_details}")

        # In case we failed to generate the assistant message and the generation prompt ids were already added to
        # input_ids, remove them from the end of input_ids
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
        self.truncate_output_ids(processing_class)
        assert (
            len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        ), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin]
    ) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
