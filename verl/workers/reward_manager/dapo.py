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

from typing import Optional

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from .naive import NaiveRewardManager


@register("dapo")
class DAPORewardManager(NaiveRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", qps: Optional[int | float] = None, max_concurrency: Optional[int] = None, timeout: Optional[int | float] = None, max_resp_len=None, overlong_buffer_cfg=None, **reward_kwargs) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key, qps, max_concurrency, timeout, **reward_kwargs)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        scores = super().compute_scores(reward_data)

        if self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = reward_data.batch["valid_response_lengths"] - expected_len
            overlong_rewards = (-torch.clamp(exceed_len, min=0) / overlong_buffer_len * self.overlong_buffer_cfg.penalty_factor).tolist()

            scores = [({**score, "score": score["score"] + overlong_reward} if isinstance(score, dict) else score + overlong_reward) for score, overlong_reward in zip(scores, overlong_rewards)]
            if self.overlong_buffer_cfg.log:
                for i, score in enumerate(scores):
                    if isinstance(score, dict):
                        score["overlong_reward"] = overlong_rewards[i]
                        score["overlong"] = overlong_rewards[i] < 0
        return scores
