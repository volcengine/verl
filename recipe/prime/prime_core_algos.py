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


def compute_rloo_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):
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

        if 'rm_scores' in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.:
            reward_tensor = data.batch['rm_scores']
            reward_mask = response_mask.bool()

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_dpo_coef)

        if 'acc' in data.batch.keys() and config.algorithm.reward_gt_coef != 0.:
            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)

            prompt_ids = data.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(-1)

            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = True
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1] = data.batch['acc']

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_gt_coef)

        final_reward_tensor = sum(reward_tensors)

        returns = (final_reward_tensor * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        advantages = returns.clone()
        advantages = verl_F.masked_whiten(advantages, response_mask)

        return advantages, returns


def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss


def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode='none'):
    # we always assume that the BoN size equals n_samples
    # mode1: use acc as rm
    # mode2: use Q as rm
    cur_Q = (token_level_scores * response_mask).sum(dim=1) * beta
    other_Q = torch.zeros_like(cur_Q)
    for i in range(token_level_scores.shape[0]):
        if acc[i] > 0:
            Q_chosen = Q_bc[i][acc_bc[i] < acc[i]]
        else:
            Q_chosen = Q_bc[i][acc_bc[i] > acc[i]]
        if len(Q_chosen) > 0:
            other_Q[i] = Q_chosen.mean() * beta
        else:
            other_Q[i] = 0
    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    if bon_mode == 'none':
        dpo_loss = dpo_loss.mean()
    else:
        weight = torch.zeros_like(dpo_loss)
        n_samples = acc_bc.shape[1]
        if bon_mode == 'bon_rm':
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((Q_bc[i] * beta <= cur_Q[i]).float().mean(), n_samples - 1)
        elif bon_mode == 'bon_acc':
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((acc_bc[i] <= acc[i]).float().mean(), n_samples - 1)
        else:
            raise NotImplementedError
        dpo_loss = (dpo_loss * weight).sum()

    return dpo_loss


def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id:start_id + n_samples] *
                      response_mask[start_id:start_id + n_samples]).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id:start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        if cur_acc_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() *
                       cur_acc_diff.abs()).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()


def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
