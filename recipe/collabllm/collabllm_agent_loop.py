# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

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
from copy import deepcopy
from typing import Any
from uuid import uuid4

from recipe.collabllm.utils import is_valid_messages
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.schemas import Message

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CollabLLMAgentLoop(ToolAgentLoop):
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        # for collabllm, firstly generate model reponses
        await self._handle_pending_state(agent_data, sampling_params)

        status = await self._handle_generating_state(agent_data, sampling_params)

        if status == AgentState.TERMINATED:
            # tell reward manager to score -1 and skip future interaction
            # to avoid reward hacking with incompleted message
            num_repeats = 0
        else:
            # then, collect interaction rollouts
            num_repeats = self.config.actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts

        interaction_requests = [deepcopy(agent_data) for _ in range(num_repeats)]

        # messages are only used in collabllm reward manager
        messages_lst = []
        for _agent_data in interaction_requests:
            if not is_valid_messages(_agent_data.messages[-1]):
                break

            prev_msg_len = len(_agent_data.messages)
            await self.run_agent_data_loop(_agent_data, sampling_params, AgentState.INTERACTING)
            messages_lst.append([Message(**msg) for msg in _agent_data.messages])

            if interaction.config.get("enable_log"):
                print(f"Assistant: ...{messages_lst[-1][prev_msg_len - 1].content[-100:]}")
                print(f"User:      {messages_lst[-1][prev_msg_len].content[:100]}...")

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={
                "turn_scores": agent_data.turn_scores,
                "messages": {"messages": messages_lst},  # compatiable with sglang interaction
            },
        )
        return output

    async def run_agent_data_loop(self, agent_data: AgentData, sampling_params: dict[str, Any], state: AgentState):
        """
        Run the agent data loop to process the agent data.

        Args:
            agent_data (AgentData): The agent data to process.
            sampling_params (dict[str, Any]): The sampling parameters.
            state (AgentState, optional): The initial state of the agent. Defaults to None.
        """

        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
