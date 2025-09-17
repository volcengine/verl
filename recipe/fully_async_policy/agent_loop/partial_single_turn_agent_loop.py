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
from typing import Any, Optional
from uuid import uuid4

from recipe.fully_async_policy.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("partial_single_turn_agent")
class PartialSingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        param_version: int,
        output: Optional[AgentLoopOutput],
    ) -> AgentLoopOutput:
        metrics = {}
        param_version_start = None
        param_version_end = None
        if not output:
            prompt_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            )
            param_version_start = param_version
        else:
            if output.is_cancel:
                # Resume the paused sample,
                # add the result directly after prompt_ids,
                # and reset generate_sequences metric
                prompt_ids = output.prompt_ids + output.response_ids
                metrics["generate_sequences"] = output.metrics.generate_sequences
                param_version_start = output.param_version_start
            else:
                # In the same batch of samples,
                # ome are canceled and some are not.
                # The samples without partial rollout are returned directly.
                return output
        param_version_end = param_version
        request_id = uuid4().hex
        with simple_timer("generate_sequences", metrics):
            response_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )

        if not output:
            response_mask = [1] * len(response_ids)
        else:
            # Pause the sample to be resumed, add the output result to response_ids, and reset response_mask
            prompt_ids = output.prompt_ids
            log_probs = output.log_probs + log_probs
            response_ids = output.response_ids + response_ids
            response_mask = [1] * len(response_ids)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
            is_cancel=is_cancel,
            log_probs=log_probs,
            param_version_start=param_version_start,
            param_version_end=param_version_end,
        )
