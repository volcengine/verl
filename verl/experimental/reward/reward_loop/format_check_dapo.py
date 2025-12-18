# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import numpy as np

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.utils.format_reward import apply_format_reward_to_score
from verl.experimental.reward.reward_loop.dapo import DAPORewardLoopManager


@register("format_check_dapo")
class FormatCheckDAPORewardLoopManager(DAPORewardLoopManager):
    """Reward loop that validates assistant message formatting and adds it to DAPO reward."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)

    async def run_single(self, data: DataProto) -> dict:
        base_result = await super().run_single(data)
        return apply_format_reward_to_score(data[0], base_result)
