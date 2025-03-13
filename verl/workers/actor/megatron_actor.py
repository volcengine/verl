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
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import importlib
from functools import partial
from packaging.version import Version
from typing import Iterable, Dict

import torch
from torch import nn
import torch.distributed
from megatron.core.optimizer import OptimizerConfig
from megatron.core import parallel_state as mpu
from megatron.core import ModelParallelConfig
from verl.utils.megatron_utils import get_model_config
from megatron.core.pipeline_parallel import get_forward_backward_func

from megatron.core.distributed import finalize_model_grads
# from megatron.core.optimizer import DistributedOptimizer

from megatron.core.optimizer import DistributedOptimizer

from omegaconf import OmegaConf
from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss, vocab_parallel_log_probs_from_logits
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean, broadcast_dict_tensor, split_dict_tensor_into_batches

__all__ = ['MegatronPPOActor']


class MegatronPPOActor(BasePPOActor):

    def __init__(self, config, model_config, megatron_config: ModelParallelConfig, actor_module: nn.ModuleList,
                 actor_optimizer: DistributedOptimizer, actor_optimizer_config: OptimizerConfig):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            megatron_config (OmegaConf): megatron configuration. It must contains

                ``sequence_parallel_enabled``: whether the sequence parallel is enabled.

                ``param_dtype``: the dtype of the parameters.

                ``virtual_pipeline_model_parallel_size``: virtual pipeline model parallel size. a.k.a number of chunks in each pp stage.
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation. Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron. It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> def megatron_actor_model_provider(pre_process, post_process):
        >>>     vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        >>>     parallel_model = ParallelMistralForCausalLMRmPadPP(config=actor_model_config,
        >>>                                                        megatron_config=megatron_config,
        >>>                                                        pre_process=pre_process,
        >>>                                                        post_process=post_process).cuda()
        >>>     return parallel_model
        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          megatron_config=megatron_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.megatron_config = megatron_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.actor_optimizer_config = actor_optimizer_config

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

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get('ulysses_sequence_parallel_size', 1) == 1

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        data.batch = data.batch.contiguous()

        def compute_logprobs_fn(output, data):
            response = data['responses']
            response_length = response.size(1)
            logits = output['logits']
            logits = logits[:, -response_length - 1:-1].contiguous()
            log_probs = vocab_parallel_log_probs_from_logits(logits, response)
            return {'log_probs': log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be handled by user outside
        recompute_old_log_prob = self.config.get('recompute_old_log_prob', True)

        if recompute_old_log_prob or 'old_log_probs' not in data.batch.keys():
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch['input_ids']
            batch_size = input_ids.size(0)
            response = batch['responses']
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(data, forward_only=True, post_process_fn=compute_logprobs_fn)
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat([o['log_probs'] for o in output], dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                else:
                    log_probs = torch.empty(size=(batch_size, response_length),
                                            dtype=torch.float32,
                                            device=input_ids.device)

                # broadcast across pp ranks
                torch.distributed.broadcast(tensor=log_probs,
                                            src=mpu.get_pipeline_model_parallel_last_rank(),
                                            group=mpu.get_pipeline_model_parallel_group(),
                                            async_op=False)

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return log_probs

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def forward_backward_batch(self, data: DataProto, forward_only=False, post_process_fn=None):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)

        if data.meta_info.get('micro_batch_size', None) is not None:
            batch_size = data.meta_info['micro_batch_size']
        else:
            batch_size = self.config.ppo_micro_batch_size_per_gpu
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
        # compute input shapes for pp stages
        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                'sequence_parallel': self.megatron_config.sequence_parallel,
                'hidden_size': self.model_config.hidden_size
            })
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            if forward_only:
                if post_process_fn is None:
                    return 1.0, {'logits': output.logits}
                else:
                    return 1.0, post_process_fn(output, data)

            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            old_log_prob = data['old_log_probs']
            advantages = data['advantages']

            clip_ratio = meta_info['clip_ratio']
            entropy_coeff = meta_info['entropy_coeff']

            # compute policy loss
            logits = output.logits
            logits = logits[:, -response_length - 1:-1].contiguous()
            logits_back = logits.clone()
            log_prob = vocab_parallel_log_probs_from_logits(logits, responses)
            logits = logits_back
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                          log_prob=log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio)
            entropy_loss = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask)
            policy_loss = pg_loss - entropy_loss * entropy_coeff

            metrics = {}
            if self.config.use_kl_loss:
                ref_log_prob = data['ref_log_prob']
                # compute kl loss
                kld = core_algos.kl_penalty(logprob=log_prob,
                                            ref_logprob=ref_log_prob,
                                            kl_penalty=self.config.kl_loss_type)
                kl_loss = masked_mean(kld, response_mask)

                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                metrics['actor/kl_loss'] = kl_loss.detach().item()
                metrics['actor/kl_coef'] = self.config.kl_loss_coef

            # return loss and stats
            stats = {
                'actor/entropy_loss': entropy_loss.detach().item(),
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item()
            }
            append_to_dict(stats, metrics)
            return policy_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            position_ids = batch['position_ids']
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            if forward_only:
                meta_info = None
            else:
                meta_info = {'clip_ratio': self.config.clip_ratio, 'entropy_coeff': self.config.entropy_coeff}
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=batch_size * seq_len,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        return losses_reduced

    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        for data in dataloader:
            # data = data.batch.to(self.actor_module.device)
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            metric_micro_batch = self.forward_backward_batch(data)
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return metrics
