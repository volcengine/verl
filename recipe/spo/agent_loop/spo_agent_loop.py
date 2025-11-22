# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications Copyright 2025 SPO authors
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
SPO Agent Loop - Extends base agent loop with code generation support.

This module inherits from verl.experimental.agent_loop and only overrides
the generate_sequences method to add SPO-specific stop tokens for code generation.
"""

import asyncio

import numpy as np
import ray

from verl import DataProto

# Re-export all base classes for backward compatibility
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorkerBase as BaseAgentLoopWorkerBase,
)
from verl.experimental.agent_loop.agent_loop import get_trajectory_info
from verl.utils.transferqueue_utils import tqbridge

__all__ = [
    "AgentLoopWorkerBase",
    "SPOAgentLoopWorker",
    "SPOAgentLoopManager",
]


class AgentLoopWorkerBase(BaseAgentLoopWorkerBase):
    """SPO-specific agent loop worker with code generation stop tokens.

    Inherits all functionality from base AgentLoopWorkerBase and only overrides
    the generate_sequences method to add SPO-specific parameters:
    - stop="</code>" for code block termination
    - include_stop_str_in_output=True to include the stop token
    """

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop with SPO-specific stop tokens.

        Override: Adds stop="</code>" and include_stop_str_in_output=True
        to sampling_params for SPO code generation use case.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout

        # SPO-specific: Add stop tokens for code generation
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
            stop="</code>",                        # SPO-SPECIFIC
            include_stop_str_in_output=True,       # SPO-SPECIFIC
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output


@ray.remote
class SPOAgentLoopWorker(AgentLoopWorkerBase):
    """SPO Agent Loop Worker as a Ray remote actor.

    This is a Ray remote actor wrapper around AgentLoopWorkerBase,
    enabling distributed execution with SPO-specific stop tokens.
    """

    def __init__(self, config, server_handles, reward_router_address=None):
        """Initialize SPO Agent Loop Worker.

        Args:
            config: trainer config.
            server_handles: OpenAI compatible LLM server actor handles.
            reward_router_address: reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)


class SPOAgentLoopManager(AgentLoopManager):
    """SPO-specific Agent Loop Manager that uses SPO's AgentLoopWorker.

    Inherits all functionality from base AgentLoopManager and only overrides
    the agent_loop_workers_class to use SPOAgentLoopWorker which includes
    code generation stop tokens.
    """

    def __init__(self, config, worker_group=None, rm_wg=None):
        """Initialize SPO Agent Loop Manager.

        Args:
            config: trainer config.
            worker_group: ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_wg: Reward model worker group.
        """
        # Set SPO-specific worker class before calling parent __init__
        self.agent_loop_workers_class = SPOAgentLoopWorker
        super().__init__(config, worker_group, rm_wg)
