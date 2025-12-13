# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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
import logging
import os
import random
from typing import Any, cast
from uuid import uuid4

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from .trajectory import Trajectory

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LightningAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._validate_config(config)
        cls.agent_client_config = OmegaConf.load(config.lightning_trainer.agent_client_config_path)
        logger.info(f"LightningAgentLoop using agent_server_addr: {config.lightning_trainer.agent_server_addr}")
        cls.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls._class_initialized = True

    @classmethod
    def _validate_config(cls, config: DictConfig):
        assert config.get("lightning_trainer") is not None, "config.lightning_trainer is required"
        assert config.lightning_trainer.model_name
        assert config.lightning_trainer.agent_server_addr
        assert config.lightning_trainer.agent_client_config_path

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        model_name = self.config.lightning_trainer.model_name
        client = hydra.utils.instantiate(
            self.agent_client_config,
            server_address=self.config.lightning_trainer.agent_server_addr,
        )

        async def _wait_random(min_seconds: int = 0, max_seconds: int = 3):
            wait_time = random.uniform(min_seconds, max_seconds)
            await asyncio.sleep(wait_time)

        trace_id = str(uuid4())
        resp = None
        try:
            await _wait_random()  # avoid large amount of simultaneous requests
            logger.debug(f"AgentClient sending request {trace_id=}, {sampling_params=}")
            resp = await client.chat(
                trace_id=trace_id, sampling_params=sampling_params, max_turns=self.max_turns, **kwargs
            )
            logger.debug(f"AgentClient final response {trace_id=}: {resp}")
        except Exception as e:
            import traceback

            # client.chat should not raise exception
            logger.error(f"Error in client.chat, should not happen: {e}")
            traceback.print_exc()

        llm_router = ray.get_actor("LLMRouter")  # get LLMRouter handler by name
        assert llm_router is not None, "LLMRouter actor not found"
        trajactory = await llm_router.retrieve_trajectory.remote(model_name=model_name, trace_id=trace_id)
        logger.debug(f"Retrieved trajectory for {trace_id=}: {trajactory}")

        output = None
        if trajactory is None:
            logger.error(f"Trajectory not found for model: {model_name}, trace_id: {trace_id}")
        try:
            trajactory = cast(Trajectory, trajactory)
            output = _trajectory_to_agent_loop_output(trajactory, resp)
        except Exception as e:
            logger.error(f"Invalid trajectory for model: {model_name}, trace_id: {trace_id}, error: {e}")
        if output is None:
            output = _create_empty_agent_loop_output(
                trace_id=trace_id,
                model_name=model_name,
                prompt_length=self.config.actor_rollout_ref.rollout.prompt_length,
                response_length=self.config.actor_rollout_ref.rollout.response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                final_response=resp,
            )

        ## maybe compute score here
        ## fill in output.reward_score and output.extra_fields["reward_extra_info"]
        return self._postprocess(output)

    def _postprocess(self, output: AgentLoopOutput) -> AgentLoopOutput:
        max_response_length = self.config.actor_rollout_ref.rollout.response_length

        output.response_ids = output.response_ids[:max_response_length]
        output.response_mask = output.response_mask[:max_response_length]
        assert len(output.response_ids) == len(output.response_mask)

        if output.response_logprobs:
            output.response_logprobs = output.response_logprobs[:max_response_length]
            assert len(output.response_ids) == len(output.response_logprobs)

        return output


def _trajectory_to_agent_loop_output(trajectory: Trajectory, final_response: Any) -> AgentLoopOutput:
    last_item = trajectory.get_last_item()
    if last_item is None:
        raise ValueError(f"Trajectory is empty, model: {trajectory.model_name}, trace_id: {trajectory.trace_id}")

    ## TODO: metrics
    output = AgentLoopOutput(
        prompt_ids=last_item.prompt_ids,
        response_ids=last_item.response_ids,
        response_mask=last_item.response_mask,
        response_logprobs=None,
        reward_score=None,
        num_turns=len(trajectory.items),
        metrics={},
        extra_fields={
            "model_name": trajectory.model_name,
            "trace_id": trajectory.trace_id,
            "final_response": final_response,
        },
    )
    return output


def _create_empty_agent_loop_output(
    trace_id: str, model_name: str, prompt_length: int, response_length: int, pad_token_id: int, final_response: Any
) -> AgentLoopOutput:
    """Create an empty AgentLoopOutput, with padding response_ids and response_mask."""
    return AgentLoopOutput(
        prompt_ids=[pad_token_id] * prompt_length,
        response_ids=[pad_token_id] * response_length,
        response_mask=[0] * response_length,
        response_logprobs=None,
        reward_score=None,
        num_turns=0,
        metrics={},
        extra_fields={
            "model_name": model_name,
            "trace_id": trace_id,
            "final_response": final_response,
        },
    )
