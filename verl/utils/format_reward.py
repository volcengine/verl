# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import re
from typing import Any

import numpy as np
from verl import DataProto

__all__ = [
    "compute_format_reward",
    "apply_format_reward_to_score",
    "apply_format_reward_to_tensor",
]

# Note:
# "You can define what the 'assistant' message format should be by modifying the patterns and logic in this file.
# For example, the current implementation enforces that each assistant message must match a sequence of tags such as
# <thinking>...</thinking> followed by either <tool_call>...</tool_call> or <answer>...</answer> (see regex patterns below).
# Adjust or extend these patterns as needed to adapt to different formatting requirements."

def _extract_messages(messages_obj: Any) -> list[dict[str, Any]] | None:
    """Normalize messages to a list of dicts."""
    if messages_obj is None:
        return None
    if isinstance(messages_obj, np.ndarray):
        try:
            messages_obj = messages_obj.item()
        except ValueError:
            messages_obj = messages_obj.tolist()
    if isinstance(messages_obj, list):
        return messages_obj
    return None


def compute_format_reward(messages_obj: Any) -> tuple[float, dict[str, Any]]:
    """Compute a format reward (+0.5 / -0.5) without attaching verbose metadata."""
    messages = _extract_messages(messages_obj)
    if messages is None:
        return -0.5, {}

    assistant_contents = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            assistant_contents.append(str(msg.get("content", "")))

    if not assistant_contents:
        return -0.5, {}

    # Allow 0, 1, or 2 line breaks between </thinking> and the subsequent tag,
    # to flexibly accept tight, single line, or double line formatting.
    tool_pattern = re.compile(
        r"^<thinking>.*?</thinking>\n{0,2}<tool_call>.*?</tool_call>$", re.DOTALL
    )
    answer_pattern = re.compile(
        r"^<thinking>.*?</thinking>\n{0,2}<answer>.*?</answer>$", re.DOTALL
    )

    for idx, content in enumerate(assistant_contents):
        is_last = idx == len(assistant_contents) - 1
        expected_pattern = answer_pattern if is_last else tool_pattern
        if not expected_pattern.match(content):
            return -0.5, {}

    return 0.5, {}


def apply_format_reward_to_score(data_item: DataProto, base_result: dict) -> dict:
    """Add format reward to a scalar reward score result."""
    messages = data_item.non_tensor_batch["tool_extra_fields"].get("messages")
    format_reward, _ = compute_format_reward(messages)

    reward_score = base_result["reward_score"] + format_reward
    reward_extra_info = dict(base_result.get("reward_extra_info", {}))
    reward_extra_info["format_reward"] = format_reward

    return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}


def apply_format_reward_to_tensor(data: DataProto, reward_tensor, reward_extra_info):
    """Add format reward to the final token reward in a tensor and log extra info."""
    if "format_reward" not in reward_extra_info:
        reward_extra_info["format_reward"] = []

    for i in range(len(data)):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

        messages = data_item.non_tensor_batch["tool_extra_fields"].get("messages")
        format_reward, _ = compute_format_reward(messages)

        if valid_response_length > 0:
            reward_tensor[i, valid_response_length - 1] += format_reward

        reward_extra_info["format_reward"].append(format_reward)

    return reward_tensor, reward_extra_info
