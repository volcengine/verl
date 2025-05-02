

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
import numpy as np
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.workers.actor import DataParallelPPOActor

import math
from collections import defaultdict

__all__ = ['DataParallelPPOActor']

class SPINDataParallelPPOActor(DataParallelPPOActor):
    
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
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

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

    def update_policy_dpo_with_ref(self, data: DataProto):
        """
        Performs the DPO update step using pre-calculated reference log probs
        from an external, periodically updated reference model.
        """
        self.actor_module.train() # Ensure training mode

        # --- Retrieve necessary data ---
        try:
            # Expects batch prepared by fit_dpo loop, including reference log probs
            batch_td = data.batch
            chosen_labels = batch_td['chosen_labels']
            rejected_labels = batch_td['rejected_labels']
            # ... other needed tensors like chosen/rejected input_ids, attention_mask, position_ids ...

            # === Get PRE-CALCULATED reference log probs from input data ===
            reference_chosen_logps = batch_td['reference_chosen_logps'] # Should be sequence-level logps
            reference_rejected_logps = batch_td['reference_rejected_logps'] # Should be sequence-level logps
            # ============================================================

            # Get DPO params from meta_info
            beta = data.meta_info.get('dpo_beta', 0.1) # Default beta
            loss_type = data.meta_info.get('dpo_loss_type', 'sigmoid')
            label_smoothing = data.meta_info.get('dpo_label_smoothing', 0.0)
            # reference_free should now be False as we provide ref logps
            reference_free = data.meta_info.get('reference_free', False) # Default False

        except KeyError as e:
            print(f"ERROR: Missing required key for DPO update (in update_policy_dpo): {e}")
            print(f"Available keys in data.batch: {list(batch_td.keys())}") # Debug print
            return {} # Return empty metrics on error
        except Exception as e_data:
            print(f"ERROR accessing data for DPO update (in update_policy_dpo): {e_data}")
            return {}

        # --- Micro-batching Setup ---
        micro_batch_size = self.config.get('ppo_micro_batch_size_per_gpu')
        if micro_batch_size is None:
            # Fallback or default if not set, or raise error
            micro_batch_size = 1 # Example fallback, adjust as needed
            print(f"Warning: 'ppo_micro_batch_size_per_gpu' not set, defaulting to {micro_batch_size}")
            # raise ValueError("Config 'ppo_micro_batch_size_per_gpu' must be set.")

        # Ensure chosen_input_ids exists before getting shape
        if 'chosen_input_ids' not in batch_td:
             print("ERROR: 'chosen_input_ids' not found in batch_td for DPO update.")
             return {}
        bsz = batch_td['chosen_input_ids'].shape[0]

        if bsz == 0:
            print("Warning: DPO batch size is 0 in update_policy_dpo. Skipping update.")
            return {'actor/dpo_loss': 0.0, 'actor/grad_norm': 0.0} # Return zero metrics if batch is empty

        num_micro_batches = math.ceil(bsz / micro_batch_size)
        gradient_accumulation_steps = num_micro_batches

        # --- Metrics Accumulation ---
        total_loss = 0.0
        accumulated_metrics = defaultdict(list)
        metrics = {} # Final metrics dict

        # --- Zero Gradients ---
        self.actor_optimizer.zero_grad(set_to_none=True)

        # --- Micro-batch Loop ---
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, bsz)
            if start_idx >= end_idx: continue

            # Slice the full DPO batch into micro-batches
            # Important: Slice ALL required tensors, including labels and inputs
            micro_batch_chosen_labels = chosen_labels[start_idx:end_idx]
            micro_batch_rejected_labels = rejected_labels[start_idx:end_idx]
            micro_batch_chosen_inputs = {
                'input_ids': batch_td['chosen_input_ids'][start_idx:end_idx],
                'attention_mask': batch_td['chosen_attention_mask'][start_idx:end_idx]
            }
            if 'chosen_position_ids' in batch_td:
                micro_batch_chosen_inputs['position_ids'] = batch_td['chosen_position_ids'][start_idx:end_idx]

            micro_batch_rejected_inputs = {
                'input_ids': batch_td['rejected_input_ids'][start_idx:end_idx],
                'attention_mask': batch_td['rejected_attention_mask'][start_idx:end_idx]
            }
            if 'rejected_position_ids' in batch_td:
                micro_batch_rejected_inputs['position_ids'] = batch_td['rejected_position_ids'][start_idx:end_idx]


            # Determine autocast dtype
            autocast_dtype = torch.bfloat16 # Or get dynamically from config/FSDP settings
            # --- Autocast Forward Pass ---
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                # --- Step 1: Forward pass for CURRENT policy log probs (with grad) ---
                policy_chosen_outputs = self.actor_module(**micro_batch_chosen_inputs, use_cache=False)
                policy_rejected_outputs = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)

                # --- Step 2: Calculate CURRENT policy log probs using get_batch_logps ---
                policy_chosen_logps = get_batch_logps(
                    policy_chosen_outputs.logits, micro_batch_chosen_labels, average_log_prob=False
                )
                policy_rejected_logps = get_batch_logps(
                    policy_rejected_outputs.logits, micro_batch_rejected_labels, average_log_prob=False
                )

                # --- Step 3: Retrieve PRE-CALCULATED reference log probs (NO grad needed) ---
                # Slice the full batch reference logps for the current micro-batch
                micro_ref_chosen_logps = reference_chosen_logps[start_idx:end_idx]
                micro_ref_rejected_logps = reference_rejected_logps[start_idx:end_idx]
                # --- The ActorAsRef calculation block is REMOVED ---

                # --- Step 4: Calculate DPO Logits and Loss ---
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = micro_ref_chosen_logps - micro_ref_rejected_logps # Uses pre-calculated values
                logits = pi_logratios - ref_logratios # DPO logits

                loss = compute_online_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,         # Has grad
                    policy_rejected_logps=policy_rejected_logps,       # Has grad
                    reference_chosen_logps=micro_ref_chosen_logps,   # No grad (from input)
                    reference_rejected_logps=micro_ref_rejected_logps, # No grad (from input)
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    reference_free=reference_free # Should be False now
                )

                # --- Scale loss for gradient accumulation ---
                scaled_loss = loss / gradient_accumulation_steps

                # --- Accumulate Metrics ---
                total_loss += loss.item() # Unscaled loss
                accumulated_metrics['actor/dpo_loss_batch'].append(loss.item())
                accumulated_metrics['actor/dpo_logits_batch'].append(logits.mean().item())
                # Accumulate policy and reference log probs/ratios if needed for debugging
                accumulated_metrics['actor/policy_chosen_logps_batch'].append(policy_chosen_logps.mean().item())
                accumulated_metrics['actor/policy_rejected_logps_batch'].append(policy_rejected_logps.mean().item())
                accumulated_metrics['actor/reference_chosen_logps_batch'].append(micro_ref_chosen_logps.mean().item())
                accumulated_metrics['actor/reference_rejected_logps_batch'].append(micro_ref_rejected_logps.mean().item())

            # --- Backward Pass (outside autocast) ---
            # Check if loss requires grad before backward
            if scaled_loss.requires_grad:
                scaled_loss.backward()
            else:
                print(f"Warning: Scaled loss at micro-batch {i} does not require grad. Skipping backward.")


        # --- End Micro-batch Loop ---

        # --- Optimizer Step (after accumulating gradients for all micro-batches) ---
        grad_norm = self._optimizer_step()

        # --- Populate Final Metrics ---
        if num_micro_batches > 0 and bsz > 0: # Check if any processing happened
            metrics['actor/dpo_loss'] = total_loss / num_micro_batches
            metrics['actor/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float('inf')
            # Average other accumulated metrics
            for key, val_list in accumulated_metrics.items():
                if val_list: metrics[key.replace('_batch','')] = np.mean(val_list)

            # Calculate accuracy / rewards / margins based on averaged logprobs if desired
            if 'actor/policy_chosen_logps' in metrics and 'actor/policy_rejected_logps' in metrics and \
               'actor/reference_chosen_logps' in metrics and 'actor/reference_rejected_logps' in metrics:
                policy_ratio_mean = metrics['actor/policy_chosen_logps'] - metrics['actor/policy_rejected_logps']
                ref_ratio_mean = metrics['actor/reference_chosen_logps'] - metrics['actor/reference_rejected_logps']
                logits_mean = policy_ratio_mean - ref_ratio_mean
                metrics['actor/rewards_chosen'] = beta * (metrics['actor/policy_chosen_logps'] - metrics['actor/reference_chosen_logps'])
                metrics['actor/rewards_rejected'] = beta * (metrics['actor/policy_rejected_logps'] - metrics['actor/reference_rejected_logps'])
                metrics['actor/rewards_accuracies'] = float(logits_mean > 0) # Mean accuracy proxy
                metrics['actor/rewards_margins'] = metrics['actor/rewards_chosen'] - metrics['actor/rewards_rejected']

        else: # Handle case where no micro-batches were run (e.g., bsz=0)
             metrics['actor/dpo_loss'] = 0.0
             metrics['actor/grad_norm'] = 0.0
             # Initialize other metrics to 0 or NaN as appropriate
             for key in accumulated_metrics.keys():
                 metrics[key.replace('_batch','')] = 0.0
             metrics['actor/rewards_chosen'] = 0.0
             metrics['actor/rewards_rejected'] = 0.0
             metrics['actor/rewards_accuracies'] = 0.0
             metrics['actor/rewards_margins'] = 0.0


        return metrics # Return aggregated metrics

# class DataParallelPPOActor(BasePPOActor):

#     def __init__(
#         self,
#         config,
#         actor_module: nn.Module,
#         actor_optimizer: torch.optim.Optimizer = None,
#     ):
#         """When optimizer is None, it is Reference Policy"""
#         super().__init__(config)
#         self.actor_module = actor_module
#         self.actor_optimizer = actor_optimizer
#         self.use_remove_padding = self.config.get('use_remove_padding', False)
#         print(f'Actor use_remove_padding={self.use_remove_padding}')
#         self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
#         self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

#         self.compute_entropy_from_logits = (
#             torch.compile(verl_F.entropy_from_logits, dynamic=True)
#             if self.config.get('use_torch_compile', True)  #  use torch compile by default
#             else verl_F.entropy_from_logits)

#     def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns: 
#             entropy: # (bs, response_len)
#             log_probs: # (bs, response_len)
#         """
#         response_length = micro_batch['responses'].size(-1)
#         multi_modal_inputs = {}
#         if 'multi_modal_inputs' in micro_batch:
#             for key in micro_batch['multi_modal_inputs'][0].keys():
#                 multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
#                                                     dim=0)

#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             input_ids = micro_batch['input_ids']
#             batch_size, seqlen = input_ids.shape
#             attention_mask = micro_batch['attention_mask']
#             position_ids = micro_batch['position_ids']
#             if position_ids.dim() == 3:  # qwen2vl mrope
#                 position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

#             if self.use_remove_padding:
#                 input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
#                                                            attention_mask)  # input_ids_rmpad (total_nnz, ...)
#                 input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

#                 # unpad the position_ids to align the rotary
#                 if position_ids.dim() == 3:
#                     position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
#                                                           indices).transpose(0, 1).unsqueeze(
#                                                               1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
#                 else:
#                     position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
#                                                           indices).transpose(0, 1)

#                 # for compute the log_prob
#                 input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

#                 # pad and slice the inputs if sp > 1
#                 if self.use_ulysses_sp:
#                     input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
#                                                                                                 position_ids_rmpad, \
#                                                                                                 sp_size=self.ulysses_sequence_parallel_size)
#                     input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
#                                                                                 self.ulysses_sequence_parallel_size)

#                 input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

#                 # only pass input_ids and position_ids to enable flash_attn_varlen
#                 output = self.actor_module(input_ids=input_ids_rmpad,
#                                            attention_mask=None,
#                                            position_ids=position_ids_rmpad,
#                                            **multi_modal_inputs,
#                                            use_cache=False)  # prevent model thinks we are generating
#                 logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

#                 logits_rmpad.div_(temperature)

#                 # compute entropy
#                 entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

#                 # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
#                 log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

#                 # gather log_prob if sp > 1
#                 if self.use_ulysses_sp:
#                     # gather and unpad for the ulysses sp
#                     log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
#                     entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
#                                                             gather_dim=0,
#                                                             unpad_dim=0,
#                                                             padding_size=pad_size)
#                 # pad back to (bsz, seqlen)
#                 full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
#                                          indices=indices,
#                                          batch=batch_size,
#                                          seqlen=seqlen)
#                 full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
#                                            indices=indices,
#                                            batch=batch_size,
#                                            seqlen=seqlen)

#                 # only return response part:
#                 entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
#                 log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

#             else:  # not using rmpad and no ulysses sp
#                 output = self.actor_module(input_ids=input_ids,
#                                            attention_mask=attention_mask,
#                                            position_ids=position_ids,
#                                            **multi_modal_inputs,
#                                            use_cache=False)  # prevent model thinks we are generating
#                 logits = output.logits
#                 logits.div_(temperature)
#                 logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
#                 log_probs = logprobs_from_logits(logits, micro_batch['responses'])
#                 entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

#             return entropy, log_probs

#     def _optimizer_step(self):
#         assert self.config.grad_clip is not None

#         if isinstance(self.actor_module, FSDP):
#             grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
#         else:
#             grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

#         # if grad_norm is not finite, skip the update
#         if not torch.isfinite(grad_norm):
#             print(f"WARN: grad_norm is not finite: {grad_norm}")
#             self.actor_optimizer.zero_grad()
#         else:
#             self.actor_optimizer.step()
#         return grad_norm

#     def compute_log_prob(self, data: DataProto) -> torch.Tensor:
#         """Compute the log probability of the responses given input_ids, attention_mask and position_ids

#         Args:
#             data (DataProto): a DataProto containing keys

#                 ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
#                 concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

#                 ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

#                 ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

#                 ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

#         Returns:
#             torch.Tensor: the log_prob tensor
#         """
#         # set to eval
#         self.actor_module.eval()

#         micro_batch_size = data.meta_info['micro_batch_size']
#         temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
#         use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

#         select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
#         batch = data.select(batch_keys=select_keys).batch
#         has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

#         if has_multi_modal_inputs:
#             num_micro_batches = data.batch.batch_size[0] // micro_batch_size
#             non_tensor_select_keys = ['multi_modal_inputs']
#             micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
#         elif use_dynamic_bsz:
#             # split using dynamic bsz
#             max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
#             micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
#         else:
#             micro_batches = batch.split(micro_batch_size)

#         log_probs_lst = []
#         for micro_batch in micro_batches:
#             if isinstance(micro_batch, DataProto):
#                 micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

#             with torch.no_grad():
#                 _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
#             log_probs_lst.append(log_probs)
#         log_probs = torch.concat(log_probs_lst, dim=0)

#         if use_dynamic_bsz:
#             indices = list(itertools.chain.from_iterable(indices))
#             assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
#             revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
#             log_probs = log_probs[revert_indices]

#         return log_probs

#     def update_policy(self, data: DataProto):
#         # make sure we are in training mode
#         self.actor_module.train()

#         temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

#         select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
#         if self.config.use_kl_loss:
#             select_keys.append('ref_log_prob')
#         batch = data.select(batch_keys=select_keys).batch
#         has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

#         # Split to make minibatch iterator for updating the actor
#         # See PPO paper for details. https://arxiv.org/abs/1707.06347
#         if has_multi_modal_inputs:
#             num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
#             non_tensor_select_keys = ['multi_modal_inputs']
#             dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
#         else:
#             dataloader = batch.split(self.config.ppo_mini_batch_size)

#         metrics = {}
#         for epoch in range(self.config.ppo_epochs):
#             for batch_idx, data in enumerate(dataloader):
#                 # split batch into micro_batches
#                 mini_batch = data
#                 if has_multi_modal_inputs:
#                     self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
#                     num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
#                     micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
#                 elif self.config.use_dynamic_bsz:
#                     max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
#                     micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
#                 else:
#                     self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
#                     # split batch into micro_batches
#                     micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

#                 self.actor_optimizer.zero_grad()

#                 for data in micro_batches:
#                     # Support all hardwares
#                     if isinstance(data, DataProto):
#                         data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
#                     else:
#                         data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
#                     responses = data['responses']
#                     response_length = responses.size(1)
#                     attention_mask = data['attention_mask']
#                     response_mask = attention_mask[:, -response_length:]
#                     old_log_prob = data['old_log_probs']
#                     advantages = data['advantages']

#                     clip_ratio = self.config.clip_ratio
#                     clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
#                     clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
#                     clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
#                     entropy_coeff = self.config.entropy_coeff
#                     loss_agg_mode = self.config.loss_agg_mode

#                     # all return: (bsz, response_length)
#                     entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

#                     pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
#                         old_log_prob=old_log_prob,
#                         log_prob=log_prob,
#                         advantages=advantages,
#                         response_mask=response_mask,
#                         cliprange=clip_ratio,
#                         cliprange_low=clip_ratio_low,
#                         cliprange_high=clip_ratio_high,
#                         clip_ratio_c=clip_ratio_c)
#                     # compute entropy loss from entropy
#                     entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

#                     # compute policy loss
#                     policy_loss = pg_loss - entropy_loss * entropy_coeff

#                     if self.config.use_kl_loss:
#                         ref_log_prob = data['ref_log_prob']
#                         # compute kl loss
#                         kld = kl_penalty(logprob=log_prob,
#                                          ref_logprob=ref_log_prob,
#                                          kl_penalty=self.config.kl_loss_type)
#                         kl_loss = agg_loss(loss_mat=kld,
#                                            loss_mask=response_mask,
#                                            loss_agg_mode=self.config.loss_agg_mode)

#                         policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
#                         metrics['actor/kl_loss'] = kl_loss.detach().item()
#                         metrics['actor/kl_coef'] = self.config.kl_loss_coef

#                     if self.config.use_dynamic_bsz:
#                         # relative to the dynamic bsz
#                         loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
#                     else:
#                         loss = policy_loss / self.gradient_accumulation
#                     loss.backward()

#                     data = {
#                         'actor/entropy': entropy_loss.detach().item(),
#                         'actor/pg_loss': pg_loss.detach().item(),
#                         'actor/pg_clipfrac': pg_clipfrac.detach().item(),
#                         'actor/ppo_kl': ppo_kl.detach().item(),
#                         'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
#                     }
#                     append_to_dict(metrics, data)

#                 grad_norm = self._optimizer_step()
#                 data = {'actor/grad_norm': grad_norm.detach().item()}
#             append_to_dict(metrics, data)
#         self.actor_optimizer.zero_grad()
#         return metrics
    
#     def update_policy_dpo(self, data: DataProto):
#         """
#         Performs the DPO update step using the provided chosen/rejected data.
#         Calculates log probs based on the CURRENT model state (ActorAsRef).
#         """
#         self.actor_module.train() # Ensure training mode

#         # --- Retrieve necessary data ---
#         try:
#             # Expects batch prepared by fit_online_dpo loop
#             batch_td = data.batch
#             chosen_labels = batch_td['chosen_labels']
#             rejected_labels = batch_td['rejected_labels']
#             # ... other needed tensors like chosen/rejected input_ids, attention_mask, position_ids ...

#             # Get DPO params from meta_info
#             beta = data.meta_info.get('dpo_beta', 10)
#             loss_type = data.meta_info.get('dpo_loss_type', 'sigmoid')
#             label_smoothing = data.meta_info.get('dpo_label_smoothing', 0.0)
#             # Note: reference_free should be True when calling compute_online_dpo_loss in ActorAsRef
#             reference_free = data.meta_info.get('reference_free', True)

#         except KeyError as e:
#             print(f"ERROR: Missing required key for DPO update (in update_policy_dpo): {e}")
#             return {} # Return empty metrics on error
#         except Exception as e_data:
#             print(f"ERROR accessing data for DPO update (in update_policy_dpo): {e_data}")
#             return {}

#         # --- Micro-batching Setup (Similar to PPO update_policy) ---
#         micro_batch_size = self.config.get('ppo_micro_batch_size_per_gpu')
#         if micro_batch_size is None:
#             raise ValueError("Config 'ppo_micro_batch_size_per_gpu' must be set.")
#         bsz = batch_td['chosen_input_ids'].shape[0]
#         num_micro_batches = math.ceil(bsz / micro_batch_size)
#         gradient_accumulation_steps = num_micro_batches

#         # --- Metrics Accumulation ---
#         total_loss = 0.0
#         accumulated_metrics = defaultdict(list)
#         metrics = {} # Final metrics dict

#         # --- Zero Gradients ---
#         self.actor_optimizer.zero_grad(set_to_none=True)

#         # --- Micro-batch Loop ---
#         for i in range(num_micro_batches):
#             start_idx = i * micro_batch_size
#             end_idx = min(start_idx + micro_batch_size, bsz)
#             if start_idx >= end_idx: continue

#             # Slice the full DPO batch into micro-batches
#             micro_batch_td = batch_td[start_idx:end_idx]

#             # Determine autocast dtype (copy from your working update_policy/update_actor)
#             autocast_dtype = torch.bfloat16 # Default
#             # ... (logic to get dtype from FSDP config) ...

#             # --- Autocast Forward Pass ---
#             with torch.autocast(device_type='cuda', dtype=autocast_dtype):
#                 # --- Step 1: Forward pass for CURRENT policy log probs (with grad) ---
#                 chosen_inputs = {
#                     'input_ids': micro_batch_td['chosen_input_ids'],
#                     'attention_mask': micro_batch_td['chosen_attention_mask']
#                 }
#                 if 'chosen_position_ids' in micro_batch_td: chosen_inputs['position_ids'] = micro_batch_td['chosen_position_ids']
#                 # This forward pass uses CURRENT parameters theta and tracks gradients
#                 policy_chosen_outputs = self.actor_module(**chosen_inputs, use_cache=False)

#                 rejected_inputs = {
#                     'input_ids': micro_batch_td['rejected_input_ids'],
#                     'attention_mask': micro_batch_td['rejected_attention_mask']
#                 }
#                 if 'rejected_position_ids' in micro_batch_td: rejected_inputs['position_ids'] = micro_batch_td['rejected_position_ids']
#                 # This forward pass also uses CURRENT theta and tracks gradients
#                 policy_rejected_outputs = self.actor_module(**rejected_inputs, use_cache=False)

#                 # --- Step 2: Calculate CURRENT policy log probs using get_batch_logps ---
#                 # These tensors WILL require grad w.r.t theta
#                 policy_chosen_logps = get_batch_logps(
#                     policy_chosen_outputs.logits, micro_batch_td['chosen_labels'], average_log_prob=False
#                 )
#                 policy_rejected_logps = get_batch_logps(
#                     policy_rejected_outputs.logits, micro_batch_td['rejected_labels'], average_log_prob=False
#                 )

#                 # --- Step 3: Calculate reference log probs (ActorAsRef - no grad) ---
#                 with torch.no_grad():
#                     # Use the SAME logits from above, but detached from grad graph
#                     reference_chosen_logps = get_batch_logps(
#                         policy_chosen_outputs.logits.detach(), micro_batch_td['chosen_labels'], average_log_prob=False
#                     )
#                     reference_rejected_logps = get_batch_logps(
#                         policy_rejected_outputs.logits.detach(), micro_batch_td['rejected_labels'], average_log_prob=False
#                     )

#                 # --- Step 4: Calculate DPO Logits and Loss ---
#                 pi_logratios = policy_chosen_logps - policy_rejected_logps
#                 # ref_logratios is calculated using detached logits, so it doesn't contribute to gradient w.r.t theta
#                 ref_logratios = reference_chosen_logps - reference_rejected_logps
#                 logits = pi_logratios - ref_logratios # DPO logits

#                 loss = compute_online_dpo_loss(
#                     policy_chosen_logps=policy_chosen_logps, # Has grad
#                     policy_rejected_logps=policy_rejected_logps, # Has grad
#                     reference_chosen_logps=reference_chosen_logps, # No grad
#                     reference_rejected_logps=reference_rejected_logps, # No grad
#                     beta=beta,
#                     label_smoothing=label_smoothing,
#                     loss_type=loss_type,
#                     reference_free=reference_free # Should be True for ActorAsRef
#                 )
#                 # The loss tensor now depends on policy_chosen_logps and policy_rejected_logps,
#                 # which depend on the forward passes that require grad.

#                 # --- Scale loss for gradient accumulation ---
#                 scaled_loss = loss / gradient_accumulation_steps

#                 # --- Accumulate Metrics ---
#                 total_loss += loss.item() # Unscaled loss
#                 # ... (accumulate other metrics like logits, logprobs etc. into accumulated_metrics dict) ...
#                 accumulated_metrics['actor/dpo_loss_batch'].append(loss.item())
#                 accumulated_metrics['actor/dpo_logits_batch'].append(logits.mean().item())


#             # --- Backward Pass (outside autocast) ---
#             # Calculates gradients based on the dependency: scaled_loss -> policy logps -> logits -> model params
#             scaled_loss.backward()

#         # --- End Micro-batch Loop ---

#         # --- Optimizer Step (after accumulating gradients for all micro-batches) ---
#         grad_norm = self._optimizer_step() # Applies update using accumulated gradients

#         # --- Populate Final Metrics ---
#         if bsz > 0: # Check if any pairs were processed
#             metrics['actor/dpo_loss'] = total_loss / num_micro_batches
#             metrics['actor/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float('inf')
#             # ... (average other accumulated metrics) ...
#             for key, val_list in accumulated_metrics.items():
#                 if val_list: metrics[key.replace('_batch','')] = np.mean(val_list)
#             # ... (calculate final reward/margin metrics based on averaged logprobs) ...
#         else:
#             metrics['actor/dpo_loss'] = 0.0; metrics['actor/grad_norm'] = 0.0

#         return metrics # Return aggregated metrics
    
#     # Inside DataParallelPPOActor class in dpo2/dp_actor.py

#     def update_policy_dpo_with_ref(self, data: DataProto):
#         """
#         Performs the DPO update step using pre-calculated reference log probs
#         from an external, periodically updated reference model.
#         """
#         self.actor_module.train() # Ensure training mode

#         # --- Retrieve necessary data ---
#         try:
#             # Expects batch prepared by fit_dpo loop, including reference log probs
#             batch_td = data.batch
#             chosen_labels = batch_td['chosen_labels']
#             rejected_labels = batch_td['rejected_labels']
#             # ... other needed tensors like chosen/rejected input_ids, attention_mask, position_ids ...

#             # === Get PRE-CALCULATED reference log probs from input data ===
#             reference_chosen_logps = batch_td['reference_chosen_logps'] # Should be sequence-level logps
#             reference_rejected_logps = batch_td['reference_rejected_logps'] # Should be sequence-level logps
#             # ============================================================

#             # Get DPO params from meta_info
#             beta = data.meta_info.get('dpo_beta', 0.1) # Default beta
#             loss_type = data.meta_info.get('dpo_loss_type', 'sigmoid')
#             label_smoothing = data.meta_info.get('dpo_label_smoothing', 0.0)
#             # reference_free should now be False as we provide ref logps
#             reference_free = data.meta_info.get('reference_free', False) # Default False

#         except KeyError as e:
#             print(f"ERROR: Missing required key for DPO update (in update_policy_dpo): {e}")
#             print(f"Available keys in data.batch: {list(batch_td.keys())}") # Debug print
#             return {} # Return empty metrics on error
#         except Exception as e_data:
#             print(f"ERROR accessing data for DPO update (in update_policy_dpo): {e_data}")
#             return {}

#         # --- Micro-batching Setup ---
#         micro_batch_size = self.config.get('ppo_micro_batch_size_per_gpu')
#         if micro_batch_size is None:
#             # Fallback or default if not set, or raise error
#             micro_batch_size = 1 # Example fallback, adjust as needed
#             print(f"Warning: 'ppo_micro_batch_size_per_gpu' not set, defaulting to {micro_batch_size}")
#             # raise ValueError("Config 'ppo_micro_batch_size_per_gpu' must be set.")

#         # Ensure chosen_input_ids exists before getting shape
#         if 'chosen_input_ids' not in batch_td:
#              print("ERROR: 'chosen_input_ids' not found in batch_td for DPO update.")
#              return {}
#         bsz = batch_td['chosen_input_ids'].shape[0]

#         if bsz == 0:
#             print("Warning: DPO batch size is 0 in update_policy_dpo. Skipping update.")
#             return {'actor/dpo_loss': 0.0, 'actor/grad_norm': 0.0} # Return zero metrics if batch is empty

#         num_micro_batches = math.ceil(bsz / micro_batch_size)
#         gradient_accumulation_steps = num_micro_batches

#         # --- Metrics Accumulation ---
#         total_loss = 0.0
#         accumulated_metrics = defaultdict(list)
#         metrics = {} # Final metrics dict

#         # --- Zero Gradients ---
#         self.actor_optimizer.zero_grad(set_to_none=True)

#         # --- Micro-batch Loop ---
#         for i in range(num_micro_batches):
#             start_idx = i * micro_batch_size
#             end_idx = min(start_idx + micro_batch_size, bsz)
#             if start_idx >= end_idx: continue

#             # Slice the full DPO batch into micro-batches
#             # Important: Slice ALL required tensors, including labels and inputs
#             micro_batch_chosen_labels = chosen_labels[start_idx:end_idx]
#             micro_batch_rejected_labels = rejected_labels[start_idx:end_idx]
#             micro_batch_chosen_inputs = {
#                 'input_ids': batch_td['chosen_input_ids'][start_idx:end_idx],
#                 'attention_mask': batch_td['chosen_attention_mask'][start_idx:end_idx]
#             }
#             if 'chosen_position_ids' in batch_td:
#                 micro_batch_chosen_inputs['position_ids'] = batch_td['chosen_position_ids'][start_idx:end_idx]

#             micro_batch_rejected_inputs = {
#                 'input_ids': batch_td['rejected_input_ids'][start_idx:end_idx],
#                 'attention_mask': batch_td['rejected_attention_mask'][start_idx:end_idx]
#             }
#             if 'rejected_position_ids' in batch_td:
#                 micro_batch_rejected_inputs['position_ids'] = batch_td['rejected_position_ids'][start_idx:end_idx]


#             # Determine autocast dtype
#             autocast_dtype = torch.bfloat16 # Or get dynamically from config/FSDP settings
#             # --- Autocast Forward Pass ---
#             with torch.autocast(device_type='cuda', dtype=autocast_dtype):
#                 # --- Step 1: Forward pass for CURRENT policy log probs (with grad) ---
#                 policy_chosen_outputs = self.actor_module(**micro_batch_chosen_inputs, use_cache=False)
#                 policy_rejected_outputs = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)

#                 # --- Step 2: Calculate CURRENT policy log probs using get_batch_logps ---
#                 policy_chosen_logps = get_batch_logps(
#                     policy_chosen_outputs.logits, micro_batch_chosen_labels, average_log_prob=False
#                 )
#                 policy_rejected_logps = get_batch_logps(
#                     policy_rejected_outputs.logits, micro_batch_rejected_labels, average_log_prob=False
#                 )

#                 # --- Step 3: Retrieve PRE-CALCULATED reference log probs (NO grad needed) ---
#                 # Slice the full batch reference logps for the current micro-batch
#                 micro_ref_chosen_logps = reference_chosen_logps[start_idx:end_idx]
#                 micro_ref_rejected_logps = reference_rejected_logps[start_idx:end_idx]
#                 # --- The ActorAsRef calculation block is REMOVED ---

#                 # --- Step 4: Calculate DPO Logits and Loss ---
#                 pi_logratios = policy_chosen_logps - policy_rejected_logps
#                 ref_logratios = micro_ref_chosen_logps - micro_ref_rejected_logps # Uses pre-calculated values
#                 logits = pi_logratios - ref_logratios # DPO logits

#                 loss = compute_online_dpo_loss(
#                     policy_chosen_logps=policy_chosen_logps,         # Has grad
#                     policy_rejected_logps=policy_rejected_logps,       # Has grad
#                     reference_chosen_logps=micro_ref_chosen_logps,   # No grad (from input)
#                     reference_rejected_logps=micro_ref_rejected_logps, # No grad (from input)
#                     beta=beta,
#                     label_smoothing=label_smoothing,
#                     loss_type=loss_type,
#                     reference_free=reference_free # Should be False now
#                 )

#                 # --- Scale loss for gradient accumulation ---
#                 scaled_loss = loss / gradient_accumulation_steps

#                 # --- Accumulate Metrics ---
#                 total_loss += loss.item() # Unscaled loss
#                 accumulated_metrics['actor/dpo_loss_batch'].append(loss.item())
#                 accumulated_metrics['actor/dpo_logits_batch'].append(logits.mean().item())
#                 # Accumulate policy and reference log probs/ratios if needed for debugging
#                 accumulated_metrics['actor/policy_chosen_logps_batch'].append(policy_chosen_logps.mean().item())
#                 accumulated_metrics['actor/policy_rejected_logps_batch'].append(policy_rejected_logps.mean().item())
#                 accumulated_metrics['actor/reference_chosen_logps_batch'].append(micro_ref_chosen_logps.mean().item())
#                 accumulated_metrics['actor/reference_rejected_logps_batch'].append(micro_ref_rejected_logps.mean().item())

#             # --- Backward Pass (outside autocast) ---
#             # Check if loss requires grad before backward
#             if scaled_loss.requires_grad:
#                 scaled_loss.backward()
#             else:
#                 print(f"Warning: Scaled loss at micro-batch {i} does not require grad. Skipping backward.")


#         # --- End Micro-batch Loop ---

#         # --- Optimizer Step (after accumulating gradients for all micro-batches) ---
#         grad_norm = self._optimizer_step()

#         # --- Populate Final Metrics ---
#         if num_micro_batches > 0 and bsz > 0: # Check if any processing happened
#             metrics['actor/dpo_loss'] = total_loss / num_micro_batches
#             metrics['actor/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float('inf')
#             # Average other accumulated metrics
#             for key, val_list in accumulated_metrics.items():
#                 if val_list: metrics[key.replace('_batch','')] = np.mean(val_list)

#             # Calculate accuracy / rewards / margins based on averaged logprobs if desired
#             if 'actor/policy_chosen_logps' in metrics and 'actor/policy_rejected_logps' in metrics and \
#                'actor/reference_chosen_logps' in metrics and 'actor/reference_rejected_logps' in metrics:
#                 policy_ratio_mean = metrics['actor/policy_chosen_logps'] - metrics['actor/policy_rejected_logps']
#                 ref_ratio_mean = metrics['actor/reference_chosen_logps'] - metrics['actor/reference_rejected_logps']
#                 logits_mean = policy_ratio_mean - ref_ratio_mean
#                 metrics['actor/rewards_chosen'] = beta * (metrics['actor/policy_chosen_logps'] - metrics['actor/reference_chosen_logps'])
#                 metrics['actor/rewards_rejected'] = beta * (metrics['actor/policy_rejected_logps'] - metrics['actor/reference_rejected_logps'])
#                 metrics['actor/rewards_accuracies'] = float(logits_mean > 0) # Mean accuracy proxy
#                 metrics['actor/rewards_margins'] = metrics['actor/rewards_chosen'] - metrics['actor/rewards_rejected']

#         else: # Handle case where no micro-batches were run (e.g., bsz=0)
#              metrics['actor/dpo_loss'] = 0.0
#              metrics['actor/grad_norm'] = 0.0
#              # Initialize other metrics to 0 or NaN as appropriate
#              for key in accumulated_metrics.keys():
#                  metrics[key.replace('_batch','')] = 0.0
#              metrics['actor/rewards_chosen'] = 0.0
#              metrics['actor/rewards_rejected'] = 0.0
#              metrics['actor/rewards_accuracies'] = 0.0
#              metrics['actor/rewards_margins'] = 0.0


#         return metrics # Return aggregated metrics