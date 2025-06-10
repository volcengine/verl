# Copyright 2025 Individual Contributor: Mert Unsal
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

import torch

from verl import DataProto
<<<<<<< HEAD
from verl.workers.reward_manager.base import BaseRewardManager


class BatchRewardManager(BaseRewardManager):
    def compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        return self.user_defined_compute_scores(
            data_sources=reward_data.non_tensor_batch["data_sources"],
            solution_strs=reward_data.non_tensor_batch["solution_strs"],
            ground_truths=reward_data.non_tensor_batch["ground_truths"],
            extra_infos=reward_data.non_tensor_batch["extra_infos"],
=======
from verl.workers.reward_manager import register


@register("batch")
class BatchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
>>>>>>> upstream/main
        )

    def postprocess_scores(self, reward_data: DataProto, training_data: DataProto) -> None:
        training_data.batch["acc"] = torch.tensor(reward_data.batch["scores"], device=training_data.batch["prompts"].device)
        super().postprocess_scores(reward_data, training_data)
