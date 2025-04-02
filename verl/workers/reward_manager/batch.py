# Copyright 2025 Mert Unsal
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


class BatchRewardManager:

    def __init__(self, tokenizer, num_examine, compute_score):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

    def verify(self, data):
        prompt_ids = data.batch['prompts']
        response_ids = data.batch['responses']
        attention_mask = data.batch['attention_mask']

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch['reward_model'].get('ground_truth', None) for item in data]
        data_sources = data.non_tensor_batch['data_source']
        extras = data.non_tensor_batch.get('extra_info', [None] * len(data))

        try:
            scores = self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
            )
        except Exception as e:
            print(f"[verify] Scoring failed: {e}")
            scores = [0.0] * len(data)

        scores = [float(s) if isinstance(s, (float, int)) else 0.0 for s in scores]
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        prompt_ids = data.batch['prompts']
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch['attention_mask']
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch['data_source']

        scores = self.verify(data)

        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            reward_tensor[i, length - 1] = scores[i]

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch['responses'][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch['prompts'][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch['reward_model'].get('ground_truth', None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        return reward_tensor