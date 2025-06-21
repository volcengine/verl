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
import logging
import os
from typing import Any, Dict, List

from verl.agent.agent_loop import AgentLoopBase, AgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

    async def run(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        completions = await self.server_manager.chat_completions(request_id=None, messages=messages, sampling_params=sampling_params)
        message = completions.choices[0].message
        messages.append(message.model_dump(exclude_unset=True, exclude_none=True))
        return messages

    def tokenize(self, messages: List[Dict[str, str]]) -> Dict[str, List]:
        prompt = self.tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
        sequence = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        response = sequence[len(prompt) :]

        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[: self.prompt_length],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=len(messages),
        )
        return output
