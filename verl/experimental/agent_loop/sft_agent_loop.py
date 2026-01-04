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
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("sft")
class SFTAgentLoop(SingleTurnAgentLoop):
    """Naive agent loop that only do single turn chat completion."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            sft_response = kwargs["sft_response"]
            sft_response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(
                    sft_response,
                ),
            )

        response_mask = [1] * len(sft_response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=sft_response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output
