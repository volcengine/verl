# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F # Import F
from megatron.core import ModelParallelConfig, tensor_parallel # Import ModelParallelConfig and tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn
from transformers import LlamaConfig
from transformers.utils import is_flash_attn_2_available

# Reuse or adapt components from parallel_attention
from .parallel_linear import QKVParallelLinear
# Use RoPE implementation compatible with potential Flash Attn usage later
from .parallel_attention import (
    LlamaRotaryEmbedding,
    repeat_kv,
    rotate_half, # Needed for manual RoPE application if flash_attn is unavailable
)
from verl.utils.megatron import tensor_parallel as tp_utils


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash

class ParallelLlamaRingAttentionRmPad(nn.Module):
    """
    Parallel Llama Attention layer with Ring Attention implementation supporting unpadded inputs.

    This layer implements Ring Attention, where the sequence is split across
    devices in the Tensor Parallel group, and Key/Value blocks are communicated
    in a ring fashion to compute attention scores distributively. It handles
    variable sequence lengths by analyzing token distribution and using temporary
    padding/unpadding internally if flash_attn's varlen interface is not available.
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.dropout_p = config.attention_dropout # Get dropout from config

        # --- Tensor Parallelism Setup ---
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()

        if self.tp_size <= 1:
             raise ValueError("RingAttention requires tensor parallel world size > 1.")

        assert self.num_heads % self.tp_size == 0, "num_heads must be divisible by tp_size"
        # QKVParallelLinear splits across the combined QKV dimension, so KV heads must also be divisible by TP size.
        assert self.num_key_value_heads % self.tp_size == 0, "num_key_value_heads must be divisible by tp_size for QKVParallelLinear"

        self.num_heads_per_tp = self.num_heads // self.tp_size
        # K/V heads are split across TP ranks in this implementation (consistent with QKVParallelLinear)
        self.num_key_value_heads_per_tp = self.num_key_value_heads // self.tp_size
        self.hidden_size_per_tp = self.hidden_size // self.tp_size

        # --- Get Parallelism Kwargs ---
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()
        if megatron_config is not None:
            if "config" not in column_kwargs: column_kwargs["config"] = megatron_config
            if "config" not in row_kwargs: row_kwargs["config"] = megatron_config
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # --- Layers ---
        # QKV Projection
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads, # Total number of Q heads
            num_key_value_heads=self.num_key_value_heads, # Total number of K/V heads
            head_dim=self.head_dim,
            bias=config.attention_bias,
            gather_output=False, # Keep QKV split across TP ranks
            skip_bias_add=False, # Bias is handled internally or by kwargs
            **column_kwargs,
        )
        # Calculate local sizes based on TP split
        self.q_size = self.num_heads_per_tp * self.head_dim
        # K and V sizes are also based on the local TP partition
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim

        self.o_proj = tensor_parallel.RowParallelLinear( # Output projection
            # Input size should be the local hidden size partition
            input_size=self.hidden_size, # RowParallelLinear expects full hidden size partitioned locally
            output_size=self.hidden_size, # Output is gathered
            bias=config.attention_bias,
            input_is_parallel=True,
            skip_bias_add=False, # Bias is handled internally or by kwargs
            **row_kwargs,
        )
        # --- RoPE ---
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # --- Ring Communication Setup ---
        self.send_rank = (self.tp_rank + 1) % self.tp_size
        self.recv_rank = (self.tp_rank - 1 + self.tp_size) % self.tp_size

        # --- Attention Scale ---
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)


    def _pad_local_data(self, unpadded_data, local_indices, global_cu_seqlens):
        """
        Analyzes local token distribution, sorts, and pads the data.
        Returns: padded_data, cu_seqlens_local, max_seqlen_local, sort_indices, inv_sort_indices, num_sequences_local
        """
        total_tokens_local = local_indices.numel()
        if total_tokens_local == 0: # Handle empty input case
             output_shape_dims = unpadded_data.shape[1:] # Assuming shape (tokens, ...)
             padded_shape = (0, 0, *output_shape_dims)
             return (torch.empty(padded_shape, dtype=unpadded_data.dtype, device=unpadded_data.device),
                     torch.tensor([0], dtype=torch.int32, device=local_indices.device), # cu_seqlens_local
                     0, # max_seqlen_local
                     torch.empty(0, dtype=torch.long, device=local_indices.device), # sort_indices
                     torch.empty(0, dtype=torch.long, device=local_indices.device), # inv_sort_indices
                     0) # num_sequences_local

        # 1. Determine global sequence ID for each local token
        global_cu_seqlens = global_cu_seqlens.to(local_indices.device)
        # Use right=True and subtract 1 to map index to sequence ID
        global_seq_ids_local = torch.searchsorted(global_cu_seqlens, local_indices, right=True) - 1

        # 2. Identify unique sequences and their lengths locally
        unique_global_seq_ids, counts = torch.unique(global_seq_ids_local, return_counts=True)
        num_sequences_local = unique_global_seq_ids.numel()
        seqlens_local = counts.to(torch.int32) # Ensure int32 for cu_seqlens

        if num_sequences_local == 0: # Should be covered by initial check, but for safety
             output_shape_dims = unpadded_data.shape[1:]
             padded_shape = (0, 0, *output_shape_dims)
             return (torch.empty(padded_shape, dtype=unpadded_data.dtype, device=unpadded_data.device),
                     torch.tensor([0], dtype=torch.int32, device=local_indices.device), 0,
                     torch.empty(0, dtype=torch.long, device=local_indices.device),
                     torch.empty(0, dtype=torch.long, device=local_indices.device), 0)

        # 3. Calculate local cumulative lengths and max length
        max_seqlen_local = torch.max(seqlens_local).item()
        cu_seqlens_local = torch.cat((torch.zeros(1, dtype=torch.int32, device=seqlens_local.device),
                                      torch.cumsum(seqlens_local, dim=0)))

        # 4. Get sorting indices to group tokens by sequence
        sort_indices = torch.argsort(global_seq_ids_local)
        inv_sort_indices = torch.argsort(sort_indices) # For unsorting later

        # 5. Sort the input data
        # Assuming unpadded_data shape: (total_tokens_local, ...)
        data_sorted = unpadded_data[sort_indices]

        # 6. Perform Padding (Scatter)
        padded_data = torch.zeros(num_sequences_local, max_seqlen_local, *unpadded_data.shape[1:],
                                  dtype=unpadded_data.dtype, device=unpadded_data.device)

        # Generate destination indices for scatter
        dest_batch_indices = torch.repeat_interleave(torch.arange(num_sequences_local, device=sort_indices.device),
                                                     seqlens_local.to(torch.long)) # repeat_interleave needs long counts
        dest_seq_indices = torch.cat([torch.arange(sl.item(), device=sort_indices.device) for sl in seqlens_local]) # Use .item() for loop

        padded_data[dest_batch_indices, dest_seq_indices] = data_sorted

        return padded_data, cu_seqlens_local, max_seqlen_local, sort_indices, inv_sort_indices, num_sequences_local

    def _unpad_local_data(self, padded_data, cu_seqlens_local, sort_indices, inv_sort_indices):
        """
        Unpads data using local sequence info and unsorts it back to original order.
        """
        num_sequences_local = padded_data.shape[0]
        if num_sequences_local == 0: # Handle empty input
             total_tokens_local = inv_sort_indices.numel() # Get expected output size
             output_shape = (total_tokens_local, *padded_data.shape[2:])
             return torch.empty(output_shape, dtype=padded_data.dtype, device=padded_data.device)


        seqlens_local = (cu_seqlens_local[1:] - cu_seqlens_local[:-1]).to(torch.long) # Ensure long for arange/cat

        # 1. Perform Unpadding (Gather)
        dest_batch_indices = torch.repeat_interleave(torch.arange(num_sequences_local, device=sort_indices.device),
                                                     seqlens_local)
        dest_seq_indices = torch.cat([torch.arange(sl.item(), device=sort_indices.device) for sl in seqlens_local]) # Use .item() for loop

        unpadded_sorted_data = padded_data[dest_batch_indices, dest_seq_indices]

        # 2. Unsort the data
        # Ensure inv_sort_indices is on the correct device
        inv_sort_indices = inv_sort_indices.to(unpadded_sorted_data.device)
        unpadded_data = unpadded_sorted_data[inv_sort_indices]

        return unpadded_data

    # Optional: Helper for manual attention masking if flash_attn is not available
    def _create_attention_mask(self, num_seqs_q, max_len_q, num_seqs_k, max_len_k,
                               cu_seqlens_q, cu_seqlens_k, ring_iter, device, dtype):
        # Create sequence ID masks
        q_seq_ids = torch.repeat_interleave(torch.arange(num_seqs_q, device=device),
                                            (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.long))
        k_seq_ids = torch.repeat_interleave(torch.arange(num_seqs_k, device=device),
                                            (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.long))

        # Create position-within-sequence masks
        q_pos = torch.cat([torch.arange(sl.item(), device=device) for sl in (cu_seqlens_q[1:] - cu_seqlens_q[:-1])])
        k_pos = torch.cat([torch.arange(sl.item(), device=device) for sl in (cu_seqlens_k[1:] - cu_seqlens_k[:-1])])

        # Expand to broadcastable shape for attention weights (bs_q, 1, max_len_q, max_len_k)
        # Note: This assumes padded attention calculation. Mask needs adjustment if using varlen.
        mask = torch.zeros(num_seqs_q, 1, max_len_q, max_len_k, device=device, dtype=torch.bool)

        # 1. Padding mask for K
        k_padding_mask = torch.arange(max_len_k, device=device)[None, :] >= (cu_seqlens_k[1:] - cu_seqlens_k[:-1])[:, None]
        mask |= k_padding_mask.unsqueeze(1).unsqueeze(2) # (bs_k, 1, 1, max_len_k) -> broadcast

        # 2. Padding mask for Q
        q_padding_mask = torch.arange(max_len_q, device=device)[None, :] >= (cu_seqlens_q[1:] - cu_seqlens_q[:-1])[:, None]
        mask |= q_padding_mask.unsqueeze(1).unsqueeze(-1) # (bs_q, 1, max_len_q, 1) -> broadcast

        # 3. Causal mask (only for self-attention, ring_iter == 0)
        if ring_iter == 0:
            # Ensure Q and K seq IDs match for causal mask application
            # This requires careful broadcasting if num_seqs_q != num_seqs_k
            if num_seqs_q == num_seqs_k: # Simplest case
                 causal_mask_ = torch.logical_and(
                     q_pos.unsqueeze(-1) < k_pos.unsqueeze(-2), # Q pos < K pos
                     q_seq_ids.unsqueeze(-1) == k_seq_ids.unsqueeze(-2) # Same sequence
                 )
                 # Need to scatter this back into the padded mask shape
                 # This part is complex. A simpler approach for padded is:
                 causal_mask_padded = torch.triu(torch.ones(max_len_q, max_len_k, device=device, dtype=torch.bool), diagonal=1)
                 mask |= causal_mask_padded.unsqueeze(0).unsqueeze(0) # Apply to all sequences if bs_q==bs_k
            else:
                 # Handling causal mask when bs_q != bs_k requires more complex logic
                 print("WARN: Causal masking for manual attention with differing local batch sizes not fully implemented.")
                 pass # Potentially over-masking or under-masking

        # 4. Ring mask (mask future blocks)
        elif ring_iter > 0:
            source_rank = (self.tp_rank - ring_iter + self.tp_size) % self.tp_size
            if source_rank > self.tp_rank: # K/V block is from a "future" rank
                mask |= True # Mask everything

        return mask


    def forward(
        self,
        hidden_states: torch.Tensor, # Shape: (total_tokens_local, 1, hidden_size_per_tp)
        position_ids: Optional[torch.LongTensor] = None, # Shape: (batch_size, max_seq_len) - Global positions
        sequence_length: int = None, # Max sequence length in the batch (global) - DEPRECATED if using cu_seqlens
        indices: torch.Tensor = None, # Shape: (total_tokens_global) - Maps token index to (batch_idx * seqlen + seq_idx)
        cu_seqlens: torch.Tensor = None, # Shape: (batch_size + 1) - Cumulative sequence lengths (global)
        max_seqlen_in_batch: int = None, # Max sequence length in the current batch (global)
        indices_local: Optional[torch.Tensor] = None, # Derived local indices
        total_tokens_local: Optional[int] = None, # Derived local token count
    ) -> torch.Tensor: # Output shape: (total_tokens_local, 1, hidden_size_per_tp)

        # --- 0. Input Validation and Local Index Derivation ---
        if cu_seqlens is None or indices is None:
            raise ValueError("`cu_seqlens` and global `indices` must be provided for unpadded input.")

        if total_tokens_local is None:
            total_tokens_local = hidden_states.shape[0]

        if indices_local is None:
            # Derive indices_local based on sequence parallelism logic.
            sp_world_size = mpu.get_tensor_model_parallel_world_size()
            sp_rank = mpu.get_tensor_model_parallel_rank()
            total_tokens_global = indices.shape[0]

            if total_tokens_global % sp_world_size != 0:
                 raise ValueError(
                     f"Total global tokens ({total_tokens_global}) is not divisible by "
                     f"sequence parallel world size ({sp_world_size}). "
                     "Padding might be missing upstream."
                 )

            calculated_total_local = total_tokens_global // sp_world_size
            if calculated_total_local != total_tokens_local:
                 raise ValueError(
                     f"Calculated local tokens ({calculated_total_local}) does not match "
                     f"hidden_states shape ({total_tokens_local}). Mismatch in SP logic?"
                 )

            start_idx = sp_rank * total_tokens_local
            end_idx = start_idx + total_tokens_local
            indices_local = indices[start_idx:end_idx]

        # Use max_seqlen_in_batch if provided, otherwise calculate from cu_seqlens
        if max_seqlen_in_batch is None:
             max_seqlen_in_batch = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # --- 1. QKV Projection ---
        # Input: (total_tokens_local, 1, hidden_size) -> QKVParallelLinear handles the TP split internally
        # Output: (total_tokens_local, 1, q_size + k_size + v_size) where sizes are local TP partitions
        qkv_output_packed, _ = self.qkv_proj(hidden_states)
        qkv_output_packed = qkv_output_packed.squeeze(1) # -> (total_tokens_local, q_local + k_local + v_local)
        # Split the packed output based on local sizes (Q, K, V are all split by TP)
        query_states, key_states, value_states = qkv_output_packed.split(
             [self.q_size, self.k_size, self.v_size], dim=-1
        )

        # Reshape for RoPE/Attention: (total_tokens_local, num_heads, head_dim)
        query_states = query_states.view(total_tokens_local, self.num_heads_per_tp, self.head_dim)
        key_states = key_states.view(total_tokens_local, self.num_key_value_heads_per_tp, self.head_dim)
        value_states = value_states.view(total_tokens_local, self.num_key_value_heads_per_tp, self.head_dim)


        # --- 2. Apply RoPE ---
        # RoPE needs global position IDs mapped to local tokens
        if position_ids is None:
             raise ValueError("`position_ids` (global) must be provided for RoPE.")

        if is_flash_attn_2_available():
            # Calculate local sequence info for initial Q/K needed for flash RoPE
            (_, cu_seqlens_local_init, max_seqlen_local_init, _, _, _) = \
                self._pad_local_data(query_states, indices_local, cu_seqlens) # Use query_states arbitrarily

            # Generate cos/sin cache
            cos, sin = self.rotary_emb(value_states, seq_len=max_seqlen_in_batch)
            # Flash RoPE needs half dim: (seq_len, rotary_dim / 2)
            cos = cos[:, :cos.shape[-1] // 2]
            sin = sin[:, :sin.shape[-1] // 2]

            # Apply RoPE using flash_attn function
            query_states = apply_rotary_emb_flash(
                query_states, cos, sin, interleaved=False, inplace=False,
                cu_seqlens=cu_seqlens_local_init, max_seqlen=max_seqlen_local_init
            )
            key_states = apply_rotary_emb_flash(
                key_states, cos, sin, interleaved=False, inplace=False,
                cu_seqlens=cu_seqlens_local_init, max_seqlen=max_seqlen_local_init # Use same local info
            )
        else:
            # Fallback: Manual RoPE application (requires padding/unpadding or careful indexing)
            raise NotImplementedError("Manual RoPE application for unpadded Ring Attention is complex and not implemented as fallback.")

        # --- 3. Prepare for Ring Attention ---
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output_accumulator = torch.zeros_like(query_states)

        # --- Pre-calculate Padding Info for Query (Local) ---
        # This only needs to be done once as Q is local and fixed for the loop.
        # We need this primarily for the fallback manual attention path.
        (q_padded,
         cu_seqlens_q_local,
         max_seqlen_q_local,
         sort_indices_q,
         inv_sort_indices_q,
         num_seqs_q_local) = self._pad_local_data(query_states, indices_local, cu_seqlens)

        # --- Ring Communication Buffers ---
        key_recv_buffer = torch.empty_like(key_states)
        value_recv_buffer = torch.empty_like(value_states)
        indices_recv_buffer = torch.empty_like(indices_local)

        # --- 4. Ring Attention Loop ---
        current_key_states = key_states
        current_value_states = value_states
        current_indices = indices_local

        for ring_iter in range(self.tp_size):
            # --- a. Communication ---
            recv_op_k = dist.P2POp(dist.irecv, key_recv_buffer, self.recv_rank, group=self.tp_group)
            recv_op_v = dist.P2POp(dist.irecv, value_recv_buffer, self.recv_rank, group=self.tp_group)
            recv_op_idx = dist.P2POp(dist.irecv, indices_recv_buffer, self.recv_rank, group=self.tp_group)
            send_op_k = dist.P2POp(dist.isend, current_key_states, self.send_rank, group=self.tp_group)
            send_op_v = dist.P2POp(dist.isend, current_value_states, self.send_rank, group=self.tp_group)
            send_op_idx = dist.P2POp(dist.isend, current_indices, self.send_rank, group=self.tp_group)

            ops = [recv_op_k, recv_op_v, recv_op_idx, send_op_k, send_op_v, send_op_idx]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            # --- b. Attention Calculation ---
            # Use flash_attn varlen interface if available
            if is_flash_attn_2_available():
                # Calculate local sequence info for the current K/V block
                (_, cu_seqlens_k_local, max_seqlen_k_local, _, _, _) = \
                    self._pad_local_data(current_key_states, current_indices, cu_seqlens)

                # Call flash_attn varlen function
                attn_out_unpadded = flash_attn_varlen_func(
                    query_states, # Unpadded Q (total_tokens_local, num_q, dim)
                    current_key_states, # Unpadded K (total_tokens_block, num_kv, dim)
                    current_value_states, # Unpadded V (total_tokens_block, num_kv, dim)
                    cu_seqlens_q=cu_seqlens_q_local,
                    cu_seqlens_k=cu_seqlens_k_local,
                    max_seqlen_q=max_seqlen_q_local,
                    max_seqlen_k=max_seqlen_k_local,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=(ring_iter == 0),
                )
            else:
                # Fallback to manual attention with padding
                print("WARN: Using manual attention fallback in RingAttention.")
                # Pad current K/V block
                (k_padded, cu_seqlens_k_local, max_seqlen_k_local, _, _, num_seqs_k_local) = \
                    self._pad_local_data(current_key_states, current_indices, cu_seqlens)
                (v_padded, _, _, _, _, _) = \
                    self._pad_local_data(current_value_states, current_indices, cu_seqlens)

                # Repeat K/V heads if using GQA
                if self.num_key_value_groups > 1:
                    k_padded = repeat_kv(k_padded, self.num_key_value_groups)
                    v_padded = repeat_kv(v_padded, self.num_key_value_groups)

                # Permute for BMM: (batch, num_heads, seqlen, dim)
                q_perm = q_padded.permute(0, 2, 1, 3)
                k_perm = k_padded.permute(0, 2, 1, 3)
                v_perm = v_padded.permute(0, 2, 1, 3)

                attn_weights = torch.matmul(q_perm, k_perm.transpose(2, 3)) * self.softmax_scale

                # Apply Masking
                attn_mask = self._create_attention_mask(
                     num_seqs_q_local, max_seqlen_q_local,
                     num_seqs_k_local, max_seqlen_k_local,
                     cu_seqlens_q_local, cu_seqlens_k_local,
                     ring_iter, attn_weights.device, attn_weights.dtype
                 )
                attn_weights.masked_fill_(attn_mask, torch.finfo(attn_weights.dtype).min)

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_perm.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_p, training=self.training) # Add dropout
                attn_output_block_padded = torch.matmul(attn_weights, v_perm) # (bs_q, num_q, seqlen_q, dim)
                attn_output_block_padded = attn_output_block_padded.permute(0, 2, 1, 3) # -> (bs_q, seqlen_q, num_q, dim)

                # Unpad and Unsort using Q's info
                attn_out_unpadded = self._unpad_local_data(attn_output_block_padded,
                                                           cu_seqlens_q_local,
                                                           sort_indices_q,
                                                           inv_sort_indices_q)

            # --- c. Accumulate Result ---
            attn_output_accumulator += attn_out_unpadded

            # --- d. Update K/V and indices for next iteration ---
            current_key_states = key_recv_buffer
            current_value_states = value_recv_buffer
            current_indices = indices_recv_buffer

        attn_output = attn_output_accumulator # Shape: (total_tokens_local, num_heads_per_tp, head_dim)
        attn_output = attn_output.reshape(total_tokens_local, self.hidden_size_per_tp)
        output, bias = self.o_proj(attn_output) # Bias is likely None or added inside
        output = output.unsqueeze(1) # -> (total_tokens_local, 1, hidden_size)
        return output # Assuming bias is handled by o_proj or not used