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
"""
Implement a multiprocess PPOCritic
"""

from functools import partial
from typing import Iterable

import torch
import torch.distributed
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import masked_mean, broadcast_dict_tensor, split_dict_tensor_into_batches
from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron.optimizer_config import OptimizerConfig

from megatron.optimizer import DistributedOptimizer
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func


class MegatronPPOCritic(BasePPOCritic):

    def __init__(self, config, model_config, megatron_config, critic_module: nn.ModuleList,
                 critic_optimizer: DistributedOptimizer, critic_optimizer_config: OptimizerConfig):
        super().__init__(config=config)

        self.model_config = model_config
        self.megatron_config = megatron_config

        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.critic_optimizer_config = critic_optimizer_config

        # we create a separate nametuple for optimizer step so that global args won't affect it.
        self.optimizer_step_args = OmegaConf.create({
            'skip_grad': None,
            'overlap_dp_param_comm': False,
            'overlap_dp_grad_comm': False,
            'gradient_accumulation_steps': 1,
            'sequence_parallel': self.megatron_config.sequence_parallel,
            'DDP_impl': 'local',
            'layernorm_allreduce_bucket_threshold': 0,
            'pipeline_model_parallel_split_rank': None,
            'reduce_grads_use_alltoall': False
        })

        if self.config.kl_ctrl.type == 'fixed':
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=self.config.kl_ctrl.kl_coef)
        elif self.config.kl_ctrl.type == 'adaptive':
            assert self.config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {self.config.kl_ctrl.horizon}'
            self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=self.config.kl_ctrl.kl_coef,
                                                           target_kl=self.config.kl_ctrl.target_kl,
                                                           horizon=self.config.kl_ctrl.horizon)
        else:
            raise NotImplementedError

    def compute_values(self, data: DataProto) -> DataProto:
        # data.batch = data.batch.to(self.critic_module.module.device)
        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.size(1)
        with torch.no_grad():
            output = self.forward_backward_batch(data=data, forward_only=True)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                values = torch.cat([o['vpreds'] for o in output], dim=0)  # (bs, seq_size, vocal_size)
                values = values.to(torch.float32)
            else:
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            # each tp ranks should contain the same value
            values = values * attention_mask
            values = values[:, -response_length - 1:-1]
            values = values.contiguous()

            # sync among pp ranks
            torch.distributed.broadcast(tensor=values,
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        group=mpu.get_pipeline_model_parallel_group())

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return values

    def compute_advantages(self, data: DataProto) -> DataProto:
        # data.batch = data.batch.to(self.critic_module.device)
        # TODO: in general, we should compute reward of ref_log_prob here
        responses = data.batch['responses']
        response_length = responses.size(1)
        token_level_rewards = data.batch['token_level_rewards']
        batch_size = data.batch.batch_size[0]
        dp_size = mpu.get_data_parallel_world_size()
        attention_mask = data.batch['attention_mask']
        eos_mask = attention_mask[:, -response_length:]

        # compute kl between ref_policy and current policy
        if 'ref_log_prob' in data.batch.keys():
            kld = core_algos.kl_penalty(data.batch['old_log_probs'],
                                        data.batch['ref_log_prob'],
                                        kl_penalty=self.config.kl_ctrl.kl_penalty_type)  # (batch_size, response_length)
            kld = kld * eos_mask
            beta = self.kl_ctrl.value
            rewards = token_level_rewards - beta * kld

            # per token kld
            current_kl = masked_mean(kld, mask=eos_mask).item()
            # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
            self.kl_ctrl.update(current_kl=current_kl, n_steps=batch_size * dp_size)
        else:
            beta = 0
            current_kl = 0
            rewards = token_level_rewards

        values = data.batch['values']

        gamma = self.config.gamma
        lam = self.config.lam

        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=rewards,
                                                                      values=values,
                                                                      eos_mask=eos_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

        sequence_reward = torch.sum(token_level_rewards, dim=-1)

        metrics = {
            'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
            'critic/rewards/max': torch.max(sequence_reward[eos_mask]).detach().item(),
            'critic/rewards/min': torch.min(sequence_reward[eos_mask]).detach().item(),
            'critic/advantages/mean': masked_mean(advantages, eos_mask).detach().item(),
            'critic/advantages/max': torch.max(advantages[eos_mask]).detach().item(),
            'critic/advantages/min': torch.min(advantages[eos_mask]).detach().item(),
            'critic/returns/mean': masked_mean(returns, eos_mask).detach().item(),
            'critic/returns/max': torch.max(returns[eos_mask]).detach().item(),
            'critic/returns/min': torch.min(returns[eos_mask]).detach().item(),
            'critic/values/mean': masked_mean(values, eos_mask).detach().item(),
            'critic/values/max': torch.max(values[eos_mask]).detach().item(),
            'critic/values/min': torch.min(values[eos_mask]).detach().item(),
            'critic/kld': current_kl,
            'critic/kl_coef': beta,
        }

        data.meta_info['metrics'] = metrics

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return data

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def forward_backward_batch(self, data: DataProto, forward_only=False):
        # broadcast from last pp rank to all other pp ranks
        data.batch = data.batch.contiguous()
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
        batches = split_dict_tensor_into_batches(data.batch, batch_size=self.config.ppo_micro_batch_size)
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        # compute input shapes for pp stages
        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                'sequence_parallel': self.megatron_config.sequence_parallel,
                'hidden_size': self.model_config.hidden_size
            })

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            if forward_only:
                return 1.0, {'vpreds': output.logits}

            responses = data['responses']
            attention_mask = data['attention_mask']
            values = data['values']
            returns = data['returns']
            response_length = responses.size(1)

            eos_mask = attention_mask[:, -response_length:]

            cliprange_value = self.config.cliprange_value

            vpreds = output.logits  # (bs, sequence_length)
            vpreds = vpreds[:, -response_length - 1:-1]

            vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                 values=values,
                                                                 returns=returns,
                                                                 eos_mask=eos_mask,
                                                                 cliprange_value=cliprange_value)
            stats = {
                'critic/vf_loss': vf_loss.detach().item(),
                'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
            }

            return vf_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            position_ids = batch['position_ids']
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            return output, partial(loss_func, data=batch, meta_info={})

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.critic_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                input_shapes=input_shapes,  # must set for flash-attn sequence packing
                seq_length=self.config.ppo_micro_batch_size * seq_len,  # no use when input_shapes was set
                hidden_size=self.model_config.hidden_size,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.critic_module,
                num_microbatches=n_micro_batch,
                seq_length=self.config.ppo_micro_batch_size * seq_len,  # in use for pp = 1
                hidden_size=self.model_config.hidden_size,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        return losses_reduced

    def update_critic(self, dataloader: Iterable[DataProto]):
        metrics = {}

        for data in dataloader:
            # data = data.batch.to(self.critic_module.device)
            self.critic_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.critic_module:
                chunk.zero_grad_buffer(zero_buffer=(not self.critic_optimizer_config.use_distributed_optimizer))

            metric_micro_batch = self.forward_backward_batch(data)

            update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step(
                self.megatron_config, self.megatron_config.timers)
            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        # add empty cache after each compute
        torch.cuda.empty_cache()
        return metrics
