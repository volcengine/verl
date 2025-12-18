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
from verl.utils.format_reward import compute_format_reward
from verl.experimental.reward.reward_loop.dapo import DAPORewardLoopManager


@register("format_check_dapo")
class FormatCheckDAPORewardLoopManager(DAPORewardLoopManager):
    """Reward loop that validates assistant message formatting and adds it to DAPO reward."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)

    async def run_single(self, data: DataProto) -> dict:
        base_result = await super().run_single(data)
        
        data_item = data[0]
        messages = data_item.non_tensor_batch["tool_extra_fields"].get("messages")

        format_reward, _ = compute_format_reward(messages)

        reward_score = base_result["reward_score"] + format_reward
        reward_extra_info = dict(base_result.get("reward_extra_info", {}))
        reward_extra_info["format_reward"] = format_reward

        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}
