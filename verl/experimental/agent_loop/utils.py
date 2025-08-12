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

from typing import Any, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from verl.protocol import DataProto


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""


def agent_loop_perf(metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
    timing = {}
    t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
    t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
    timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
    timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
    timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
    timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
    timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
    timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

    # batch sequence generation is bounded by the slowest sample
    slowest = np.argmax(t_generate_sequences + t_tool_calls)
    attention_mask = output.batch["attention_mask"][slowest]
    prompt_length = output.batch["prompts"].shape[1]
    timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
    timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
    timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
    timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

    return timing
