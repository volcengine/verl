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
from verl.workers.reward_manager.base import BaseRewardManager


class BatchRewardManager(BaseRewardManager):
    def compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        return self.user_defined_compute_scores(
            data_sources=reward_data.non_tensor_batch["data_sources"],
            solution_strs=reward_data.non_tensor_batch["solution_strs"],
            ground_truths=reward_data.non_tensor_batch["ground_truths"],
            extra_infos=reward_data.non_tensor_batch["extra_infos"],
        )

    def postprocess_scores(self, reward_data: DataProto, training_data: DataProto) -> None:
        training_data.batch["acc"] = torch.tensor(reward_data.batch["scores"], device=training_data.batch["prompts"].device)
        super().postprocess_scores(reward_data, training_data)
