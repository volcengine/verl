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

import itertools
import logging
import os

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch):
        if 'left_pad_len' in micro_batch.keys():
            response_length = micro_batch['input_ids'].size(-1) - micro_batch['left_pad_len'] - micro_batch['actual_prompt_len']
        else:
            response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                # NOTE(HanlinDu): why we need padding here?
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                if 'left_pad_len' in micro_batch.keys():
                    new_values = torch.zeros_like(values)
                    for idx, r in enumerate(response_length):
                        new_values[idx, :r] = values[idx, -r - 1: -1]
                    # FIXME (HanlinDu): should not truncate the values here for temporarily fixing the bug
                    # values = new_values
                    values = new_values[:, :r]
                else:
                    values = values[:, -response_length - 1:-1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length - 1 : -1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if 'window_rounds' in data.batch.keys():
            select_keys += ['left_pad_len', 'actual_prompt_len', 'window_rounds']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]
        
        response_mask = data.batch["response_mask"]
        if 'left_pad_len' in data.batch.keys():
            print(" --- use left pad --- ")
            pad_len = values.shape[-1] - response_mask.shape[-1]
            # pad from right
            response_mask = F.pad(response_mask, (0, pad_len), value=0)
        values = values * response_mask  # Only action tokens have values
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]

        use_window_rollout = 'window_rounds' in data.batch.keys()
        if use_window_rollout:
            select_keys += ['left_pad_len', 'actual_prompt_len', 'is_finished', 'rounds_eos_mask']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                ###############################################################
                if use_window_rollout:
                    mini_batch_mask, new_batch_size = [], 0
                    for item in mini_batch:
                        if item['is_finished']:
                            new_batch_size += 1
                            mini_batch_mask.append(True)
                        else:
                            mini_batch_mask.append(False)
                    if dist.is_initialized():
                        new_batch_sizes = torch.tensor([new_batch_size], device='cuda')
                        dist.all_reduce(new_batch_sizes, op=dist.ReduceOp.MAX, group=None)
                        new_batch_sizes = new_batch_sizes.cpu().item()
                    else:
                        new_batch_sizes = new_batch_size
                    dp_size = torch.distributed.get_world_size() // self.config.tp_size // self.config.ulysses_sequence_parallel_size
                    total_num = (new_batch_sizes // dp_size + 1 if new_batch_sizes % dp_size else new_batch_sizes // dp_size) * dp_size
                    if total_num == 0:
                        continue
                    elif total_num != new_batch_size:
                        for i, m in enumerate(mini_batch_mask):
                            if (not m) and new_batch_size < total_num:
                                mini_batch_mask[i] = True
                                new_batch_size += 1
                    mini_batch_mask = torch.tensor(mini_batch_mask)
                    mini_batch = mini_batch.masked_select(mini_batch_mask)
                ###################################################################
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # critic device is cpu when using offload
                    responses = data["responses"]
                    attention_mask = data["attention_mask"]
                    values = data["values"]
                    returns = data["returns"]
                    response_length = responses.size(1)

                    response_mask = attention_mask[:, -response_length:] # TODO (zht) maybe no use
                    overlong_mask = data.get('overlong_mask', None) if not use_window_rollout else data.get('is_finished', None)
                    if 'rounds_eos_mask' in data.keys():
                        eos_mask = data['rounds_eos_mask'] # TODO(zht):confirm eos_mask
                    else:
                        eos_mask = attention_mask[:, -response_length - 1:-1]
                    
                    vpreds = self._forward_micro_batch(data)
                    
                    if returns.size(-1) < vpreds.size(-1):
                        vpreds = vpreds[:, :returns.size(-1)]
                    if returns.size(-1) < values.size(-1):
                        values = values[:, :returns.size(-1)]
                    cliprange_value_low = self.config.get('cliprange_value_low', self.config.cliprange_value)
                    cliprange_value_high = self.config.get('cliprange_value_high', self.config.cliprange_value)
                    from recipe.tppo.tppo_algos import compute_value_loss

                    vf_loss, vf_clipfrac = compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        eos_mask=eos_mask,
                        cliprange_value_low=cliprange_value_low,
                        cliprange_value_high=cliprange_value_high,
                        overlong_mask=overlong_mask
                        )

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    data = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        "critic/vpred_mean": masked_mean(vpreds, eos_mask).detach().item(),
                    }

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
