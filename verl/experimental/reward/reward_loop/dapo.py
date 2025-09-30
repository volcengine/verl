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
import inspect
import logging
import os

from verl import DataProto

from .base import RewardLoopBase

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DAPORewardLoop(RewardLoopBase):
    def __init__(self, config, tokenizer, compute_score, reward_model=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)

        assert inspect.iscoroutinefunction(self.compute_score), (
            f"compute_score must be a coroutine function, but got {self.compute_score}"
        )

        overlong_buffer_cfg = config.reward_model.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward_model.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_model = reward_model
        self.reward_model_tokenizer = reward_model_tokenizer

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    async def run_single(self, data: DataProto) -> dict:
        data_source = data.non_tensor_batch["data_source"].tolist()[0]
        response_ids = data.batch["responses"].tolist()[0]
        ground_truth = data.non_tensor_batch["reward_model"].tolist()[0]["ground_truth"]
        extra_info = data.non_tensor_batch["extra_info"].tolist()[0]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        result = await self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_model=self.reward_model,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            response_length = len(response_ids)
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
