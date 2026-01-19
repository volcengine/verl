# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import json
from typing import Any, Optional

import numpy as np
import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class MultiTurnSFTDatasetDeepseek(MultiTurnSFTDataset):
    def tokenize_assistant(self, index, message, full_message, tools, enable_thinking):
        """
        reimplement the jinja logic to suit multiturn data for dsv31
        """

        has_tool_calls = "tool_calls" in message and message["tool_calls"] is not None

        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
            apply_chat_template_kwargs["thinking"] = enable_thinking

        tokens = []
        prefix = "      "

        # having tool_call and not having tool call has different processing logic
        if has_tool_calls:
            is_first = False
            for tool in message["tool_calls"]:
                formatted_args = tool["function"]["arguments"]
                if not isinstance(formatted_args, str):
                    for k, v in formatted_args.items():
                        if isinstance(v, np.ndarray):
                            formatted_args[k] = list(v)
                    formatted_args = json.dumps(formatted_args, ensure_ascii=False)

                if not is_first:
                    if message.get("content") is None:
                        tokens += processor.encode(
                            prefix
                            + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                            + tool["function"]["name"]
                            + "<｜tool▁sep｜>"
                            + formatted_args
                            + "<｜tool▁call▁end｜>",
                            add_special_tokens=False,
                        )
                    else:
                        tokens += processor.encode(
                            prefix
                            + "    "
                            + message["content"]
                            + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
                            + tool["function"]["name"]
                            + "<｜tool▁sep｜>"
                            + formatted_args
                            + "<｜tool▁call▁end｜>",
                            add_special_tokens=False,
                        )
                    is_first = True
                else:
                    tokens += processor.encode(
                        prefix
                        + "<｜tool▁call▁begin｜>"
                        + tool["function"]["name"]
                        + "<｜tool▁sep｜>"
                        + formatted_args
                        + "<｜tool▁call▁end｜>",
                        add_special_tokens=False,
                    )
            tokens += processor.encode("    <｜tool▁calls▁end｜><｜end▁of▁sentence｜>", add_special_tokens=False)
        else:
            content = message.get("content", "")
            tokens += processor.encode(prefix + content + "<｜end▁of▁sentence｜>", add_special_tokens=False)

        # construct the final input_ids
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}
        return inputs

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process a single message and return its tokenized representation.

        Args:
            index: turn index in the conversation
            message: A single message dictionary
            images: List of images to be used
            videos: List of videos to be used
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        if message["role"] == "system":
            inputs = processor.apply_chat_template(
                [message],
                tools=tools,
                # add generation prompt to True, for the '<｜Assistant｜>' token
                # Only USER will have this triggerd for dsv3
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
        elif message["role"] == "user":
            message_mod = message.copy()
            message_mod["content"] += "      <｜Assistant｜></think>"
            inputs = processor.apply_chat_template(
                [message_mod],
                # add generation prompt to True, for the '<｜Assistant｜>' token
                # Only USER will have this triggerd for dsv3
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )
        elif message["role"] == "assistant":
            inputs = self.tokenize_assistant(index, message, full_message, tools, enable_thinking)
        else:
            inputs = processor.apply_chat_template(
                [message],
                # add generation prompt to True, for the '<｜Assistant｜>' token
                # Only USER will have this triggerd for dsv3
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **apply_chat_template_kwargs,
            )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # remove system prompt if exists
        if index != 0 and message["role"] not in ["system", "assistant"]:
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs
