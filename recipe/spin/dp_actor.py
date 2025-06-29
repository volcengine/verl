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


import itertools
import math
from collections import defaultdict

import numpy as np
import torch

from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor

__all__ = ["DataParallelPPOActor"]


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

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
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
        self.actor_module.train()  # Ensure training mode

        # --- Retrieve necessary data ---
        try:
            # Expects batch prepared by fit_dpo loop, including reference log probs
            batch_td = data.batch
            chosen_labels = batch_td["chosen_labels"]
            rejected_labels = batch_td["rejected_labels"]
            # ... other needed tensors like chosen/rejected input_ids, attention_mask, position_ids ...

            # === Get PRE-CALCULATED reference log probs from input data ===
            reference_chosen_logps = batch_td["reference_chosen_logps"]  # Should be sequence-level logps
            reference_rejected_logps = batch_td["reference_rejected_logps"]  # Should be sequence-level logps
            # ============================================================

            # Get DPO params from meta_info
            # beta = data.meta_info.get('dpo_beta', 0.1) # Default beta
            beta = self.config.get("dpo_beta", 0.1)  # Default beta
            loss_type = data.meta_info.get("dpo_loss_type", "sigmoid")
            label_smoothing = data.meta_info.get("dpo_label_smoothing", 0.0)
            # reference_free should now be False as we provide ref logps
            reference_free = data.meta_info.get("reference_free", False)  # Default False

        except KeyError as e:
            print(f"ERROR: Missing required key for DPO update (in update_policy_dpo): {e}")
            print(f"Available keys in data.batch: {list(batch_td.keys())}")  # Debug print
            return {}  # Return empty metrics on error
        except Exception as e_data:
            print(f"ERROR accessing data for DPO update (in update_policy_dpo): {e_data}")
            return {}

        # --- Micro-batching Setup ---
        micro_batch_size = self.config.get("ppo_micro_batch_size_per_gpu")
        if micro_batch_size is None:
            # Fallback or default if not set, or raise error
            micro_batch_size = 1  # Example fallback, adjust as needed
            print(f"Warning: 'ppo_micro_batch_size_per_gpu' not set, defaulting to {micro_batch_size}")
            # raise ValueError("Config 'ppo_micro_batch_size_per_gpu' must be set.")

        # Ensure chosen_input_ids exists before getting shape
        if "chosen_input_ids" not in batch_td:
            print("ERROR: 'chosen_input_ids' not found in batch_td for DPO update.")
            return {}
        bsz = batch_td["chosen_input_ids"].shape[0]

        if bsz == 0:
            print("Warning: DPO batch size is 0 in update_policy_dpo. Skipping update.")
            return {"actor/dpo_loss": 0.0, "actor/grad_norm": 0.0}  # Return zero metrics if batch is empty

        num_micro_batches = math.ceil(bsz / micro_batch_size)
        gradient_accumulation_steps = num_micro_batches

        # --- Metrics Accumulation ---
        total_loss = 0.0
        accumulated_metrics = defaultdict(list)
        metrics = {}  # Final metrics dict

        # --- Zero Gradients ---
        self.actor_optimizer.zero_grad(set_to_none=True)

        # --- Micro-batch Loop ---
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, bsz)
            if start_idx >= end_idx:
                continue

            # Slice the full DPO batch into micro-batches
            # Important: Slice ALL required tensors, including labels and inputs
            micro_batch_chosen_labels = chosen_labels[start_idx:end_idx]
            micro_batch_rejected_labels = rejected_labels[start_idx:end_idx]
            micro_batch_chosen_inputs = {"input_ids": batch_td["chosen_input_ids"][start_idx:end_idx], "attention_mask": batch_td["chosen_attention_mask"][start_idx:end_idx]}
            if "chosen_position_ids" in batch_td:
                micro_batch_chosen_inputs["position_ids"] = batch_td["chosen_position_ids"][start_idx:end_idx]

            micro_batch_rejected_inputs = {"input_ids": batch_td["rejected_input_ids"][start_idx:end_idx], "attention_mask": batch_td["rejected_attention_mask"][start_idx:end_idx]}
            if "rejected_position_ids" in batch_td:
                micro_batch_rejected_inputs["position_ids"] = batch_td["rejected_position_ids"][start_idx:end_idx]

            # Determine autocast dtype
            autocast_dtype = torch.bfloat16  # Or get dynamically from config/FSDP settings
            # --- Autocast Forward Pass ---
            with torch.autocast(device_type=get_device_name(), dtype=autocast_dtype):
                # --- Step 1: Forward pass for CURRENT policy log probs (with grad) ---
                policy_chosen_outputs = self.actor_module(**micro_batch_chosen_inputs, use_cache=False)
                policy_rejected_outputs = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)

                # --- Step 2: Calculate CURRENT policy log probs using get_batch_logps ---
                policy_chosen_logps = get_batch_logps(policy_chosen_outputs.logits, micro_batch_chosen_labels, average_log_prob=False)
                policy_rejected_logps = get_batch_logps(policy_rejected_outputs.logits, micro_batch_rejected_labels, average_log_prob=False)

                # --- Step 3: Retrieve PRE-CALCULATED reference log probs (NO grad needed) ---
                # Slice the full batch reference logps for the current micro-batch
                micro_ref_chosen_logps = reference_chosen_logps[start_idx:end_idx]
                micro_ref_rejected_logps = reference_rejected_logps[start_idx:end_idx]
                # --- The ActorAsRef calculation block is REMOVED ---

                # --- Step 4: Calculate DPO Logits and Loss ---
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = micro_ref_chosen_logps - micro_ref_rejected_logps  # Uses pre-calculated values
                logits = pi_logratios - ref_logratios  # DPO logits

                loss = compute_online_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,  # Has grad
                    policy_rejected_logps=policy_rejected_logps,  # Has grad
                    reference_chosen_logps=micro_ref_chosen_logps,  # No grad (from input)
                    reference_rejected_logps=micro_ref_rejected_logps,  # No grad (from input)
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    reference_free=reference_free,  # Should be False now
                )

                # --- Scale loss for gradient accumulation ---
                scaled_loss = loss / gradient_accumulation_steps

                # --- Accumulate Metrics ---
                total_loss += loss.item()  # Unscaled loss
                accumulated_metrics["actor/dpo_loss_batch"].append(loss.item())
                accumulated_metrics["actor/dpo_logits_batch"].append(logits.mean().item())
                # Accumulate policy and reference log probs/ratios if needed for debugging
                accumulated_metrics["actor/policy_chosen_logps_batch"].append(policy_chosen_logps.mean().item())
                accumulated_metrics["actor/policy_rejected_logps_batch"].append(policy_rejected_logps.mean().item())
                accumulated_metrics["actor/reference_chosen_logps_batch"].append(micro_ref_chosen_logps.mean().item())
                accumulated_metrics["actor/reference_rejected_logps_batch"].append(micro_ref_rejected_logps.mean().item())

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
        if num_micro_batches > 0 and bsz > 0:  # Check if any processing happened
            metrics["actor/dpo_loss"] = total_loss / num_micro_batches
            metrics["actor/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float("inf")
            # Average other accumulated metrics
            for key, val_list in accumulated_metrics.items():
                if val_list:
                    metrics[key.replace("_batch", "")] = np.mean(val_list)

            # Calculate accuracy / rewards / margins based on averaged logprobs if desired
            if "actor/policy_chosen_logps" in metrics and "actor/policy_rejected_logps" in metrics and "actor/reference_chosen_logps" in metrics and "actor/reference_rejected_logps" in metrics:
                policy_ratio_mean = metrics["actor/policy_chosen_logps"] - metrics["actor/policy_rejected_logps"]
                ref_ratio_mean = metrics["actor/reference_chosen_logps"] - metrics["actor/reference_rejected_logps"]
                logits_mean = policy_ratio_mean - ref_ratio_mean
                metrics["actor/rewards_chosen"] = beta * (metrics["actor/policy_chosen_logps"] - metrics["actor/reference_chosen_logps"])
                metrics["actor/rewards_rejected"] = beta * (metrics["actor/policy_rejected_logps"] - metrics["actor/reference_rejected_logps"])
                metrics["actor/rewards_accuracies"] = float(logits_mean > 0)  # Mean accuracy proxy
                metrics["actor/rewards_margins"] = metrics["actor/rewards_chosen"] - metrics["actor/rewards_rejected"]

        else:  # Handle case where no micro-batches were run (e.g., bsz=0)
            metrics["actor/dpo_loss"] = 0.0
            metrics["actor/grad_norm"] = 0.0
            # Initialize other metrics to 0 or NaN as appropriate
            for key in accumulated_metrics.keys():
                metrics[key.replace("_batch", "")] = 0.0
            metrics["actor/rewards_chosen"] = 0.0
            metrics["actor/rewards_rejected"] = 0.0
            metrics["actor/rewards_accuracies"] = 0.0
            metrics["actor/rewards_margins"] = 0.0

        return metrics  # Return aggregated metrics
