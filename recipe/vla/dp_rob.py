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
Single Process Actor
"""

import itertools

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import BasePPOActor

__all__ = ["RobDataParallelPPOActor"]


class RobDataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        print(f"PRM use dynamic bsz={self.config.get('use_dynamic_bsz', False)}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False  # self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len

    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,),
            traj_len:
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask

    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked:
            entropy_masked:
        """
        mask_expanded = mask.unsqueeze(-1)

        log_probs_masked = torch.where(mask_expanded, log_probs, torch.zeros_like(log_probs, requires_grad=False))

        entropy_masked = torch.where(mask_expanded, entropy, torch.zeros_like(entropy, requires_grad=False))

        return log_probs_masked, entropy_masked

    def _forward_micro_batch(self, micro_batch, temperature) -> tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:

        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """

        batch_size = micro_batch["responses"].size(0)
        traj_len = micro_batch["responses"].size(1)
        tot_pad_len = micro_batch["input_ids"].size(2)

        assert all(
            micro_batch[key].size(0) == batch_size
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(1) == traj_len for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(micro_batch[key].size(2) == tot_pad_len for key in ["input_ids", "attention_mask"])

        response_length = micro_batch["responses"].size(-1)  # 7*8

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            pixel_values = micro_batch["pixel_values"]
            responses = micro_batch["responses"]

            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])

            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            if self.config.vla == "openvla-oft":
                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                )  # prevent model thinks we are generating

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[..., -256 - 64 : -64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                # assert (0<=responses<=255).all()

                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(log_probs.shape) == 2 and len(entropy.shape) == 2
                log_probs = log_probs.reshape((batch_size, traj_len * 8, 7))
                entropy = entropy.reshape((batch_size, traj_len * 8, 7))

                mask = self.generate_traj_mask(micro_batch["finish_step"], traj_len * 8)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)

                log_probs = log_probs.reshape((batch_size, traj_len * response_length))
                entropy = entropy.reshape((batch_size, traj_len * response_length))

            elif self.config.vla == "openvla":
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits

                logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                # ADD

                log_probs = log_probs.reshape(
                    (
                        batch_size,
                        traj_len,
                    )
                    + log_probs.shape[1:]
                )
                entropy = entropy.reshape(
                    (
                        batch_size,
                        traj_len,
                    )
                    + entropy.shape[1:]
                )

                mask = self.generate_traj_mask(micro_batch["finish_step"], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)

                log_probs = log_probs.reshape((batch_size, traj_len * response_length))
                entropy = entropy.reshape((batch_size, traj_len * response_length))

            return entropy, log_probs

    def _forward_micro_batch_update(
        self, input_ids, attention_mask, pixel_values, responses, temperature
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.vla == "openvla-oft":
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                )

                assert logits.requires_grad

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[..., -256 - 64 : -64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index

                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs

            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                #

                logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
                logits = logits.div(temperature)

                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs

    def _forward_micro_batch_entropy(self, micro_batch, temperature) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch["responses"].size(0)
        traj_len = micro_batch["responses"].size(1)
        tot_pad_len = micro_batch["input_ids"].size(2)

        assert all(
            micro_batch[key].size(0) == batch_size
            for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(
            micro_batch[key].size(1) == traj_len for key in ["responses", "input_ids", "attention_mask", "pixel_values"]
        )
        assert all(micro_batch[key].size(2) == tot_pad_len for key in ["input_ids", "attention_mask"])

        response_length = micro_batch["responses"].size(-1)
        # assert response_length == 7*8

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            # batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            pixel_values = micro_batch["pixel_values"]

            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])

            input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
            attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

            if self.config.vla == "openvla-oft":
                logits = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                )

                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256
                logits = logits[..., -256 - 64 : -64]  # Shape: [batch_size, seq_len, 256]

                logits = logits.div(temperature)

                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape) == 2
                entropy = entropy.reshape((batch_size, traj_len * 8, 7))
                mask = self.generate_traj_mask(micro_batch["finish_step"], traj_len * 8)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len * response_length))
                return entropy

            elif self.config.vla == "openvla":
                output = self.actor_module(
                    input_ids=input_ids_unpad,
                    attention_mask=attention_mask_unpad,
                    pixel_values=pixel_values,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                #

                logits = logits[:, -response_length - 1 : -1]  # (bsz, response_length)
                logits = logits.div(temperature)

                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                # ADD

                entropy = entropy.reshape(
                    (
                        batch_size,
                        traj_len,
                    )
                    + entropy.shape[1:]
                )
                mask = self.generate_traj_mask(micro_batch["finish_step"], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len * response_length))
                return entropy

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

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
            torch.Tensor: the log_prob tensor
        """

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]  # 256
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]  # trues
        self.pad_token_id = data.meta_info["pad_token_id"]

        select_keys = ["responses", "input_ids", "attention_mask", "pixel_values", "finish_step"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "pixel_values",
            "old_log_probs",
            "advantages",
            "finish_step",
        ]
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data["responses"]

                response_length = responses.size(1) * responses.size(2)
                finish_step = data["finish_step"] * self.config.action_token_len
                steps = torch.arange(response_length, device=data["responses"].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data["responses"].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]

                # clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data["responses"].size(0)
                traj_len = data["responses"].size(1)
                tot_pad_len = data["input_ids"].size(2)

                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                pixel_values = data["pixel_values"]
                responses = data["responses"]

                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])

                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    "actor/pg_loss": 0,
                    "actor/pg_clipfrac": 0,
                    "actor/ppo_kl": 0,
                }

                assert traj_len % self.config.traj_mini_batch_size == 0
                traj_split_num = int(traj_len / self.config.traj_mini_batch_size)

                for i in range(0, traj_len, int(traj_len / traj_split_num)):
                    entropy, log_prob = self._forward_micro_batch_update(
                        input_ids=input_ids[i : i + int(traj_len / traj_split_num)],
                        attention_mask=attention_mask[i : i + int(traj_len / traj_split_num)],
                        pixel_values=pixel_values[i : i + int(traj_len / traj_split_num)],
                        responses=responses[i : i + int(traj_len / traj_split_num)],
                        temperature=temperature,
                    )

                    slice_id = i * self.config.action_token_len * self.config.action_chunks_len
                    next_slice_id = (
                        (i + int(traj_len / traj_split_num))
                        * self.config.action_token_len
                        * self.config.action_chunks_len
                    )
                    old_log_prob_tmp = old_log_prob[:, slice_id:next_slice_id]
                    advantages_tmp = advantages[:, slice_id:next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id:next_slice_id]

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob_tmp,
                        log_prob=log_prob,
                        advantages=advantages_tmp,
                        eos_mask=response_mask_tmp,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_low=clip_ratio_low,
                    )

                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss * response_mask_tmp_sum
                    pg_clipfrac = pg_clipfrac * response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl * response_mask_tmp_sum / response_mask_sum

                    policy_loss = pg_loss / response_mask_sum

                    loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    loss_info["actor/pg_loss"] = loss_info["actor/pg_loss"] + policy_loss.detach().item()
                    loss_info["actor/pg_clipfrac"] = loss_info["actor/pg_clipfrac"] + pg_clipfrac.detach().item()
                    loss_info["actor/ppo_kl"] = loss_info["actor/ppo_kl"] + ppo_kl.detach().item()

                append_to_dict(metrics, loss_info)

            grad_norm = self._optimizer_step()
            data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    def compute_entropy(self, bacth_data: DataProto):
        if bacth_data.meta_info["train_mode"]:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = bacth_data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ["responses", "input_ids", "attention_mask", "pixel_values", "finish_step"]
        batch = bacth_data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        print("dataloader_length:", len(dataloader))

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data["responses"]
                response_length = responses.size(1) * responses.size(2)
                finish_step = data["finish_step"] * self.config.action_token_len
                steps = torch.arange(response_length, device=data["responses"].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data["responses"].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)

                with torch.no_grad():
                    entropy = self._forward_micro_batch_entropy(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                if bacth_data.meta_info["is_filtered"] and bacth_data.meta_info["train_mode"]:
                    data = {
                        "actor_after/entropy_loss_train": entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif bacth_data.meta_info["is_filtered"] and not bacth_data.meta_info["train_mode"]:
                    data = {
                        "actor_after/entropy_loss_eval": entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info["is_filtered"] and bacth_data.meta_info["train_mode"]:
                    data = {
                        "actor_before/entropy_loss_train": entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info["is_filtered"] and not bacth_data.meta_info["train_mode"]:
                    data = {
                        "actor_before/entropy_loss_eval": entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
