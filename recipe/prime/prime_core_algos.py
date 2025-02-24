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

import torch
import verl
import verl.utils.torch_functional as verl_F


def compute_rloo_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    # calculate rloo reward on different reward sources, and sum again

    def masked_rloo(reward_tensor_original, mask_tensor):
        reward_tensor = reward_tensor_original.clone()
        reward_tensor[~mask_tensor] = 0
        for start_pos in range(0, reward_tensor.shape[0], n_samples):
            cur_rewards_mean = torch.cat([
                reward_tensor[pos:pos + 1][mask_tensor[pos:pos + 1]].mean(dim=0, keepdim=True)
                for pos in range(start_pos, start_pos + n_samples)
            ],
                                         dim=0)
            cur_rewards_sum = cur_rewards_mean.sum()
            cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
            reward_tensor[start_pos:start_pos + n_samples][
                mask_tensor[start_pos:start_pos + n_samples]] = \
                reward_tensor[start_pos:start_pos + n_samples][
                    mask_tensor[start_pos:start_pos + n_samples]] * (
                        n_samples / (n_samples - 1)) - cur_reward_baseline

        return reward_tensor

    reward_tensors = []

    with torch.no_grad():

        if 'rm_scores' in data.batch and config.algorithm.dpo_coef != 0.:
            reward_tensor = data.batch['rm_scores']
            reward_mask = eos_mask.bool()

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.dpo_coef)

        if 'acc' in data.batch and config.algorithm.gt_coef != 0.:
            reward_tensor = torch.zeros_like(eos_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(eos_mask, dtype=torch.bool)

            prompt_ids = data.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)

            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = True
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = data.batch['acc']

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.gt_coef)

        final_reward_tensor = sum(reward_tensors)

        returns = (final_reward_tensor * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, eos_mask)

        return advantages, returns
