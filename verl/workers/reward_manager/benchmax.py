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

from collections import defaultdict
from pathlib import Path

import torch

from verl import DataProto
from verl.tools.utils.tool_registry import get_tool_class
from verl.workers.reward_manager import register


@register("benchmax")
class BenchmaxRewardManager:
    """BenchmaxRewardManager is a reward manager that computes rewards as defined in a benchmax environment."""

    def __init__(self, tokenizer, benchmax_cls_name="", **kwargs) -> None:
        """
        Initialize the BenchmaxRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
        """
        assert benchmax_cls_name, "Specify benchmax class name"
        self.benchmax_cls = get_tool_class(benchmax_cls_name)
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["ground_truth"]
            workspace = data_item.non_tensor_batch.get("workspaces", "")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            reward = 0
            for fn in self.benchmax_cls.reward_funcs:
                reward += fn(
                    prompt=prompt_str,
                    completion=response_str,
                    ground_truth=ground_truth,
                    workspace=Path(workspace),
                    **extra_info,
                )

            reward_tensor[i, valid_response_length - 1] = reward
            print("[response]", response_str)
            print("[ground_truth]", ground_truth)
            print("[reward]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
