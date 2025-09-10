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
import torch
from verl import DataProto
# Import the CORRECT parallel computation function
from verl.utils.reward_score.math_verify import parallel_compute_score


class ParallelRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        num_processes: int = 16,  # This now correctly controls the pool size
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = parallel_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_processes = num_processes # Set the desired level of parallelism

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Processes the data in a single parallel call."""

        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        # --- Step 1: Collect all data ---
        model_outputs = []
        ground_truths = []
        metadata = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            model_outputs.append(response_str)
            ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
            metadata.append({
                "original_index": i,
                "valid_response_length": valid_response_length,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                "data_source": data_item.non_tensor_batch[self.reward_fn_key],
            })

        # --- Step 2: Make a single, efficient parallel call ---
        # The parallel function will correctly use self.num_processes to limit concurrency.
        all_scores = self.compute_score(
            model_outputs=model_outputs,
            ground_truths=ground_truths,
            num_processes=self.num_processes
        )

        # --- Step 3: Assign rewards and handle logging ---
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i, score in enumerate(all_scores):
            meta_item = metadata[i]
            original_index = meta_item["original_index"]
            valid_response_length = meta_item["valid_response_length"]
            data_source = meta_item["data_source"]
            reward = float(score)

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            if valid_response_length > 0:
                reward_tensor[original_index, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", meta_item["prompt_str"])
                print("[response]", meta_item["response_str"])
                print("[ground_truth]", meta_item["ground_truth"])
                print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
