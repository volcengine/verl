# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import logging
import os
from typing import Iterable

import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_sppo_loss(
    old_log_prob: torch.Tensor,  # (bs, seq_len)
    log_prob: torch.Tensor,  # (bs, seq_len)
    rewards: torch.Tensor,  # (bs,)
    response_mask: torch.Tensor,  # (bs, seq_len)
    eta: float = 1.0,
    loss_agg_mode: str = "token-mean",
):
    """
    SPPO Loss computation.
    """
    # Compute log-ratios over masked tokens
    log_prob_sum = (log_prob * response_mask).sum(dim=1)  # (bs,)
    old_log_prob_sum = (old_log_prob * response_mask).sum(dim=1)  # (bs,)
    log_ratios = log_prob_sum - old_log_prob_sum  # (bs,)

    scaled_rewards = eta * (rewards)
    loss_vec = (log_ratios - scaled_rewards) ** 2  # (bs,)

    if loss_agg_mode == "token-mean":
        sample_mask = response_mask.any(dim=1).float()  # (bs,)
        loss = verl_F.masked_mean(loss_vec, sample_mask)

    return loss, log_ratios, scaled_rewards


class DataParallelSPPOActor(DataParallelPPOActor):
    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``seq_level_rewards``: tensor of shape [batch_size, ]. torch.float32. The
                seq_level_rewards of responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        multi_turn = data.meta_info.get("multi_turn", False)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "seq_level_rewards"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            non_tensor_select_keys = ["multi_modal_inputs"]
            data = data.select(select_keys, non_tensor_select_keys)
        else:
            data = data.select(batch_keys=select_keys)

        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]):
        # make sure we are in training mode
        self.actor_module.train()
        metrics = {}

        for mini_batch in dataloader:
            self.actor_optimizer.zero_grad()
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            for data in micro_batches:
                # Support all hardwares
                if isinstance(data, DataProto):
                    data = {**data.batch.to(get_device_id()), **data.non_tensor_batch, **data.meta_info}
                else:
                    data = data.to(get_device_id())  # actor device is cpu when using offload
                responses = data["responses"]
                response_length = responses.size(1)
                attention_mask = data["attention_mask"]

                if "loss_mask" in data:
                    response_mask = data["loss_mask"][:, -response_length:]
                else:
                    response_mask = attention_mask[:, -response_length:]

                old_log_prob = data["old_log_probs"]
                rewards = data["seq_level_rewards"]

                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode
                eta = self.config.get("sppo_eta", 1.0)

                # all return: (bsz, response_length)
                calculate_entropy = False
                if entropy_coeff != 0:
                    calculate_entropy = True
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, calculate_entropy=calculate_entropy)

                pg_loss, log_ratios, preference = compute_sppo_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    rewards=rewards,
                    response_mask=response_mask,
                    eta=eta,
                    loss_agg_mode=loss_agg_mode,
                )

                if entropy_coeff != 0:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss

                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    "actor/loss": loss.detach().item(),
                    "actor/log_ratio_mean": log_ratios.mean().detach().item(),
                    "actor/preference_mean": preference.mean().detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)

        return metrics
