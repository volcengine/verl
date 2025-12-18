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

from collections import defaultdict

from verl.utils.format_reward import compute_format_reward
from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager


@register("format_check_dapo")
class FormatCheckDAPORewardManager(DAPORewardManager):
    """DAPO reward manager with an extra formatting bonus/penalty."""

    def __call__(self, data, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        base_result = super().__call__(data, return_dict=True)
        reward_tensor = base_result["reward_tensor"]

        base_extra_info = base_result.get("reward_extra_info", {})
        reward_extra_info = defaultdict(list)
        for key, value in base_extra_info.items():
            reward_extra_info[key] = value

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            messages = data_item.non_tensor_batch["tool_extra_fields"].get("messages")
            format_reward, _ = compute_format_reward(messages)

            reward_tensor[i, valid_response_length - 1] += format_reward
            reward_extra_info["format_reward"].append(format_reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
