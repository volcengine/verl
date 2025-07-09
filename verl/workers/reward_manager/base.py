# Copyright 2025 Individual Contributor: Lianyu Yao
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

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class BaseRewardManager(ABC):
    """
    Base class for reward managers.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        qps: Optional[int | float] = None,
        max_concurrency: Optional[int] = None,
        timeout: Optional[int | float] = None,
        **reward_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.user_defined_compute_scores = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.qps = qps
        self.max_concurrency = max_concurrency
        self.timeout = timeout

    def preprocess_reward_data(self, data: DataProto) -> DataProto:
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        prompt_len = prompt_ids.shape[-1]
        valid_prompt_length = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
        valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

        prompt_strs = np.array([self.tokenizer.decode(prompt_ids[i][-valid_len.item() :], skip_special_tokens=True) for i, valid_len in enumerate(valid_prompt_length)])
        solution_strs = np.array([self.tokenizer.decode(response_ids[i][: valid_len.item()], skip_special_tokens=True) for i, valid_len in enumerate(valid_response_lengths)])

        return DataProto.from_single_dict(
            {
                "prompt_strs": prompt_strs,
                "response_ids": response_ids,
                "solution_strs": solution_strs,
                "data_sources": data.non_tensor_batch[self.reward_fn_key],
                "ground_truths": np.vectorize(lambda reward_model: reward_model.get("ground_truth", None))(data.non_tensor_batch["reward_model"]),
                "extra_infos": data.non_tensor_batch.get("extra_info", np.full(len(data), None, dtype=object)),
                "valid_response_lengths": valid_response_lengths,
            }
        )

    @abstractmethod
    def compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        """Calculate scores and return a list of scores or dictionaries with additional information."""
        pass

    def postprocess_scores(self, reward_data: DataProto, training_data: DataProto) -> None:
        reward_tensor = torch.zeros_like(reward_data.batch["response_ids"], dtype=torch.float32)
        batch_indices = torch.arange(len(reward_data))
        seq_indices = reward_data.batch["valid_response_lengths"] - 1
        reward_tensor[batch_indices, seq_indices] = reward_data.batch["scores"]
        reward_data.batch["reward_tensor"] = reward_tensor

    def log_data(self, reward_data: DataProto, raw_reward: list[int | float | dict]) -> None:
        already_print_data_sources = defaultdict(int)
        for i in range(len(reward_data)):
            data_item = reward_data[i]
            data_source = data_item.non_tensor_batch["data_sources"]
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", data_item.non_tensor_batch["prompt_strs"])
                print("[response]", data_item.non_tensor_batch["solution_strs"])
                print("[ground_truth]", data_item.non_tensor_batch["ground_truths"])
                if isinstance(raw_reward[i], dict):
                    for key, value in raw_reward[i].items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", raw_reward[i])

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_data = self.preprocess_reward_data(data)
        raw_rewards = self.compute_scores(reward_data)
        assert len(raw_rewards) == len(data), f"The number of rewards must match the number of data items. Got {len(raw_rewards)} rewards for {len(data)} data items."
        reward_data.batch["scores"] = torch.tensor([(raw_reward["score"] if isinstance(raw_reward, dict) else raw_reward) for raw_reward in raw_rewards], dtype=torch.float32)
        self.postprocess_scores(reward_data, data)
        self.log_data(reward_data, raw_rewards)

        if return_dict:
            reward_extra_info = defaultdict(list)
            for raw_reward in raw_rewards:
                if isinstance(raw_reward, dict):
                    for key, value in raw_reward.items():
                        reward_extra_info[key].append(value)
                else:
                    reward_extra_info["score"].append(raw_reward)

            return {
                "reward_tensor": reward_data.batch["reward_tensor"],
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_data.batch["reward_tensor"]
