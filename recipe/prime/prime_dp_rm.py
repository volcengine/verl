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
"""
Implement a multiprocess PPOCritic
"""
import itertools
from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .prime_core_algos import compute_ce_dpo_loss_rm
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPRIMERewardModel']


class DataParallelPRIMERewardModel:

    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Reward model use_remove_padding={self.use_remove_padding}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch, prompt_length):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        input_ids = micro_batch['input_ids']
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch['attention_mask']
        position_ids = micro_batch['position_ids']

        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                       attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                            self.ulysses_sequence_parallel_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            rm_output_logits = self.reward_module(input_ids=input_ids_rmpad,
                                                  attention_mask=None,
                                                  position_ids=position_ids_rmpad,
                                                  use_cache=False).logits.squeeze(
                                                      0)  # copied. I don't really know why there is a squeeze
            rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)
            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1),
                                      indices=indices,
                                      batch=batch_size,
                                      seqlen=seqlen).squeeze(-1)

        else:
            rm_output_logits = self.reward_module(input_ids=micro_batch['input_ids'],
                                                  attention_mask=micro_batch['attention_mask'],
                                                  position_ids=micro_batch['position_ids']).logits
            rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :],
                                                          dim=-1)  # (batch_size, seq_length, vocab_size)
            rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch['input_ids'][:, 1:].unsqueeze(-1)).squeeze(
                -1)  # (batch, seq_length)

        if self.ref_module is not None:
            # 不用重复remove pad，只用做好re-pad即可
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output_logits = self.ref_module(input_ids=input_ids_rmpad,
                                                        attention_mask=None,
                                                        position_ids=position_ids_rmpad,
                                                        use_cache=False).logits.squeeze(0)
                    ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits,
                                                                 labels=input_ids_rmpad_rolled)
                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels,
                                                             gather_dim=0,
                                                             unpad_dim=0,
                                                             padding_size=pad_size)
                    ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1),
                                               indices=indices,
                                               batch=batch_size,
                                               seqlen=seqlen).squeeze(-1)
                else:
                    ref_output_logits = self.ref_module(input_ids=micro_batch['input_ids'],
                                                        attention_mask=micro_batch['attention_mask'],
                                                        position_ids=micro_batch['position_ids']).logits
                    ref_log_prob = torch.nn.functional.log_softmax(ref_output_logits[:, :-1, :],
                                                                   dim=-1)  # (batch_size, seq_length, vocab_size)
                    ref_log_labels = ref_log_prob.gather(dim=-1,
                                                         index=micro_batch['input_ids'][:, 1:].unsqueeze(-1)).squeeze(
                                                             -1)  # (batch, seq_length)
        else:
            ref_log_labels = micro_batch['old_log_probs']

        num_actions = micro_batch['input_ids'].shape[-1] - prompt_length
        max_positions = micro_batch['attention_mask'][:, prompt_length:].sum(-1)

        ref_log_labels.to(rm_log_labels.dtype)
        q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]  # this is actually diff of q

        # reward computation does not need gradient. only q needs
        with torch.no_grad():

            # generalized estimation of r should go before the reward filling. r means process reward for policy model, or the advantage of reward model.
            lam = self.config.get('lambda', 0.)
            beta = self.config.model.get('beta_train', 0.05)
            if lam == 0.:
                r = q * beta
            else:
                # reward coefficient takes no effect here
                acc = micro_batch['acc']
                q_ = q * beta
                r = torch.zeros_like(q)
                # TODO: 参考implicit value model在此处的处理方式，应该是靠直接修改max_positions[0]-1位置的q为r-Q_{t-1}，后面的r全部抹0
                lastgaelam = 0
                # change the last token and mask out all paddings to make this process easier
                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, :max_positions[i] - 1].sum()
                    q_[i, max_positions[i]:] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            step_ends = []

            if self.config.prime_granularity == 'token':
                for i in range(micro_batch['input_ids'].shape[0]):
                    step_ends.append(list(range(max_positions[i])))
            elif self.config.prime_granularity == 'whole':
                for i in range(micro_batch['input_ids'].shape[0]):
                    step_ends.append([max_positions[i] - 1])
            else:
                raise NotImplementedError

            token_level_score = torch.zeros_like(q)

            for i, step_end in enumerate(step_ends):
                for j in range(len(step_end)):
                    step_range = [
                        min(step_end[j - 1] + 1, num_actions - 1) if j > 0 else 0,
                        min(num_actions - 1, step_end[j])
                    ]
                    token_level_score[i, step_range[1]] = r[i, step_range[0]:step_range[1] + 1].sum()

        return token_level_score, q

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        if self.config.prime_norm == 'batch_norm':
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'acc']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        prompt_length = data.batch['input_ids'].shape[-1] - data.batch['responses'].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
        rm_scores = torch.concat(rm_scores_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return rm_scores, {}

    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get('beta_train', 0.05)

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'acc', 'prompts']
        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()
                attention_mask = data['attention_mask']
                acc = data['acc']

                prompt_ids = data['prompts']
                prompt_length = prompt_ids.shape[-1]

                eos_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(data, prompt_length)

                rm_scores_lst.append(rm_score)

                if self.config.model.loss_type == 'ce':
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, eos_mask=eos_mask, beta=beta)
                else:
                    raise NotImplementedError

                data = {'reward_model/dpo_loss': dpo_loss.detach().item()}

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = dpo_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'reward_model/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        return rm_scores, metrics
