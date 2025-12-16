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
import copy
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))
        video_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("video", None))

        videos = None
        videos_kwargs = {}
        if video_data is not None:
            videos = []
            video_metadata = []
            for item in video_data:
                if isinstance(item, tuple) and len(item) == 2:
                    v, meta = item
                else:
                    v, meta = item, {}
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                videos.append(v)
                video_metadata.append(meta)
            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

        metrics = {}
        request_id = uuid4().hex

        # Use processor if available for multimodal support
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # vLLM's multimodal preprocessor expects *unexpanded* placeholders in the prompt
            # (e.g. "<|vision_start|><|video_pad|><|vision_end|>"), and will expand them based on mm inputs.
            # So we:
            # 1) send unexpanded prompt ids to vLLM for generation
            # 2) keep expanded prompt ids in the rollout output for training-side forward passes
            server_prompt_ids = (
                self.tokenizer(raw_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
                .squeeze(0)
                .tolist()
            )
            expanded_inputs = self.processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                videos_kwargs=videos_kwargs,
                return_tensors="pt",
            )
            prompt_ids = expanded_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            server_prompt_ids = prompt_ids

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=server_prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={k: v for k, v in {"image": image_data, "video": video_data}.items() if v is not None},
            num_turns=2,
            metrics=metrics,
        )
        return output
