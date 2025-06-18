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
Atropos Recipe Trainer
This module provides an Atropos-compatible RL trainer that handles:
- Policy weight synchronization via VERL's sharding managers
- Advantage-weighted SFT loss computation
- Integration with Atropos environments for RL training
Based on VERL's recipe pattern, similar to DAPO.
"""

import logging
import os
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import Dataset
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available

if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_ATROPOS_LOGGING_LEVEL", "WARN"))


class AtroposTrainer(FSDPSFTTrainer):
    """
    Atropos Recipe Trainer with advantage-weighted SFT support.
    
    This trainer provides the complete interface needed for Atropos integration:
    - Token-level advantage weighting
    - Policy weight synchronization (via sharding managers)
    - Compatible with VERL's distributed training infrastructure
    """

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        super().__init__(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)

        # Atropos-specific configuration
        self.use_advantage_weighting = getattr(config, "use_advantage_weighting", True)
        self.advantage_normalization = getattr(config, "advantage_normalization", "batch")  # "none", "batch", "global"
        self.advantage_clipping = getattr(config, "advantage_clipping", None)  # None or (min_val, max_val)

        if self.device_mesh.get_rank() == 0:
            print(f"Atropos Trainer initialized:")
            print(f"  - Advantage weighting: {self.use_advantage_weighting}")
            print(f"  - Advantage normalization: {self.advantage_normalization}")
            print(f"  - Advantage clipping: {self.advantage_clipping}")

    def _normalize_advantages(self, advantages: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Normalize advantages according to the configured method."""
        if self.advantage_normalization == "none":
            return advantages

        # Only consider valid tokens for normalization
        valid_advantages = advantages[loss_mask.bool()]

        if len(valid_advantages) == 0:
            return advantages

        if self.advantage_normalization == "batch":
            # Normalize by batch statistics
            mean_adv = valid_advantages.mean()
            std_adv = valid_advantages.std() + 1e-8
            advantages = (advantages - mean_adv) / std_adv

        elif self.advantage_normalization == "global":
            # Global normalization (could be extended to track running statistics)
            mean_adv = valid_advantages.mean()
            std_adv = valid_advantages.std() + 1e-8
            advantages = (advantages - mean_adv) / std_adv

        return advantages

    def _clip_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Clip advantages to prevent extreme values."""
        if self.advantage_clipping is None:
            return advantages

        min_val, max_val = self.advantage_clipping
        return torch.clamp(advantages, min=min_val, max=max_val)

    def compute_advantage_weighted_sft_loss(
        self,
        input_ids: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Core interface for Atropos integration: advantage-weighted SFT loss.
        
        Args:
            input_ids: Batch of tokens, shape (batch_size, seq_len)
            advantages: Batch of advantages, same shape as input_ids
            loss_mask: Batch of loss masks, same shape as input_ids
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len = input_ids.shape

        # Create default masks if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Prepare batch
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }

        return self._compute_advantage_weighted_loss(batch, do_backward=False)

    def _compute_advantage_weighted_loss(
        self,
        batch: dict,
        do_backward: bool = True
    ) -> torch.Tensor:
        """Compute advantage-weighted cross-entropy loss."""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare tensors
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)

        # Check if we have advantages for this batch
        has_advantages = "advantages" in batch and self.use_advantage_weighting
        if has_advantages:
            # Advantages should be same shape as loss_mask after reshaping
            advantages = batch.pop("advantages")[:, :-1].reshape(-1).to(self.device_name)
            # Normalize and clip advantages
            advantages = self._normalize_advantages(advantages, loss_mask)
            advantages = self._clip_advantages(advantages)
        else:
            # Default to uniform advantages (equivalent to standard SFT)
            advantages = torch.ones_like(loss_mask)

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                # Apply loss mask and advantage weighting
                loss = loss * loss_mask.to(loss.device) * advantages.to(loss.device)

            else:
                # Sequence parallel path with remove padding
                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )

                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)

                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                advantages = advantages.to(full_loss.device)

                # Apply loss mask and advantage weighting
                loss = full_loss * loss_mask * advantages

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Override to support advantage weighting when advantages are provided."""
        if "advantages" in batch and self.use_advantage_weighting:
            # Use advantage-weighted loss computation
            return self._compute_advantage_weighted_loss(batch, do_backward)
        else:
            # Fall back to standard SFT loss
            return super()._compute_loss_and_backward(batch, do_backward) 