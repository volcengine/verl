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
import asyncio
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List

from omegaconf import DictConfig
from openai.types.chat.chat_completion import Choice

from verl.agent.agent_loop import AgentLoopBase, AsyncLLMServerManager
from verl.tools.base_tool import initialize_tools_from_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolAgentLoop(AgentLoopBase):
    def __init__(self, config: DictConfig, server_manager: AsyncLLMServerManager):
        super().__init__(config, server_manager)

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.system_prompt = self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=False)

    async def run(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        # For multi-turn rollout, do first turn with n>=1 and then run each choice in parallel.
        completions = await self.server_manager.chat_completions(messages=messages, sampling_params=sampling_params, request_id=None)
        sampling_params["n"] = 1

        tasks = []
        for choice in completions.choices:
            tasks.append(self._run_choice(deepcopy(messages), sampling_params, choice, completions.id))
        choices = await asyncio.gather(*tasks)
        return choices

    async def _run_choice(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any], choice: Choice, request_id: str) -> List[Dict[str, Any]]:
        while True:
            message, finish_reason = choice.message, choice.finish_reason
            if message.content is None:
                message.content = ""
            if message.tool_calls:
                message.tool_calls = message.tool_calls[: self.max_parallel_calls]
            messages.append(message.model_dump(exclude_unset=True, exclude_none=True))

            # check if we reach max turns
            if self.max_turns and len(messages) >= self.max_turns:
                return messages

            # check if the model called tools
            if finish_reason != "tool_calls":
                return messages

            # call tools
            tool_calls = message.tool_calls
            tasks = []
            for tool_call in tool_calls:
                tasks.append(self._call_tool(tool_call))
            tool_responses = await asyncio.gather(*tasks)
            if any(isinstance(item, Exception) for item in tool_responses):
                return messages
            messages.extend(tool_responses)

            # resubmit completion request with tool responses
            completions = await self.server_manager.chat_completions(messages, sampling_params, request_id)
            choice, request_id = completions.choices[0], completions.id

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        """Call tool and return tool response."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        try:
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def tokenize(self, messages: List[Dict[str, str]]) -> Dict[str, List]:
        prompt_ids = self.tokenizer.apply_chat_template(messages[:1], tools=self.tool_schemas, add_generation_prompt=False, tokenize=True)

        # last message should not be tool calling
        while messages[-1]["role"] == "tool":
            messages.pop()

        response_ids, response_mask = [], []
        last = 0
        for i in range(1, len(messages)):
            # parallel tool calls are in single turn
            if messages[i]["role"] == "tool" and messages[i + 1]["role"] == "tool":
                continue
            message_str = self.tokenizer.apply_chat_template(messages[last + 1 : i + 1], add_generation_prompt=False, tokenize=False)
            # remove system prompt prefix
            message_ids = self.tokenizer.encode(message_str[len(self.system_prompt) :])
            response_ids += message_ids
            response_mask += [0 if messages[i]["role"] == "tool" else 1] * len(message_ids)

            last = i

        output = dict(
            prompt_ids=prompt_ids[: self.prompt_length],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
        )
        return output
