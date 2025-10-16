# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial
import numpy as np
from verl.workers.reward_manager import register

def parallel_compute_score(evaluation_func, response_str, ground_truth, data_sources, timeout=6, max_workers=64):

    with tqdm(total=len(response_str)) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluation_func, response_str[index], ground_truth[index], data_sources[index]): index
                for index in range(len(response_str))
            }
            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    return [results[i] for i in range(len(response_str))]

@register("yr_code")
class YRRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None, is_long_penalty=False, is_binary_reward=True, is_power4_reward=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.compute_score = partial(self.compute_score, is_long_penalty=is_long_penalty, is_binary_reward=is_binary_reward, is_power4_reward=is_power4_reward)
        
        
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [(data_item.non_tensor_batch['reward_model']['ground_truth'] if 'livecodebench' not in data_item.non_tensor_batch['data_source'] else data_item.non_tensor_batch['extra_info']) for data_item in data]
        ground_truth = [x.tolist() if isinstance(x, np.ndarray) else x for x in ground_truth]
        data_sources = data.non_tensor_batch['data_source']

        assert len(response_str) == len(ground_truth) == len(data_sources)


        scores = []
        try:
            for i in range(0, len(response_str), 1024):
                cur_response_str = response_str[i:i+1024]
                cur_ground_truth = ground_truth[i:i+1024]
                cur_data_sources = data_sources[i:i+1024]

                cur_scores = parallel_compute_score(
                        self.compute_score,
                        cur_response_str,
                        cur_ground_truth,
                        cur_data_sources,
                    )

                scores += cur_scores
            assert len(scores) == len(response_str)

        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(response_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            # reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = float(scores[i])

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", response_str[i])

        return reward_tensor
