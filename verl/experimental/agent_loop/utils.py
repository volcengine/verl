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

import numpy as np
import torch
from pydantic import BaseModel
from tensordict import TensorDict

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
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    interrupt: bool = False


def agent_loop_postprocess(
    tokenizer, inputs: list[AgentLoopOutput], max_prompt_length: int, max_response_length: int
) -> DataProto:
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts
    tokenizer.padding_side = "left"
    outputs = tokenizer.pad(
        [{"input_ids": input.prompt_ids} for input in inputs],
        padding="max_length",
        # max_length=self.config.actor_rollout_ref.rollout.prompt_length,
        max_length=max_prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # responses
    tokenizer.padding_side = "right"
    outputs = tokenizer.pad(
        [{"input_ids": input.response_ids} for input in inputs],
        padding="max_length",
        max_length=max_response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # response_mask
    outputs = tokenizer.pad(
        [{"input_ids": input.response_mask} for input in inputs],
        padding="max_length",
        max_length=max_response_length,
        return_tensors="pt",
        return_attention_mask=False,
    )
    response_mask = outputs["input_ids"]
    assert response_ids.shape == response_mask.shape, (
        f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
    )
    response_mask = response_mask * response_attention_mask

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            "responses": response_ids,  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            "position_ids": position_ids,  # [bsz, prompt_length + response_length]
        },
        batch_size=len(input_ids),
    )

    num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
    metrics = [input.metrics.model_dump() for input in inputs]
    return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns}, meta_info={"metrics": metrics})


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
