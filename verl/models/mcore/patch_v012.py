# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None
# 2. `MultimodalRotaryEmbedding.forward` in `embeddings.py` works wrong when enable context parallelism


def apply_patch():
    import torch
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.multi_latent_attention import (
        MLASelfAttention,
        apply_rotary_pos_emb,
        deprecate_inference_params,
        gather_from_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    def patch_get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        if self.config.rope_type == "rope":
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # hidden_states:[s, b, 2048], q: [s, b, n * 192]
                q, _ = self.linear_q_proj(q_compressed)

            q_len, bsz, _ = q.size()

            # q: [s, b, n, 192]
            q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [s, b, 2048]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [s, b, n, 256]
            kv = kv.view(
                q_len,
                bsz,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            else:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length.
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # [s, b, 64] -> [s, b, 1, 64]
            k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

            # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                q_pos_emb = q_pos_emb.squeeze(1)
                k_pos_emb = k_pos_emb.squeeze(1)
                q_no_pe = q_no_pe.squeeze(1)
                k_no_pe = k_no_pe.squeeze(1)
                value = value.squeeze(1)
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
            )

            # query: [s, b, n, 192]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
            if packed_seq_params is not None:
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
            else:
                # key: [s, b, n, 192]
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value

    MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors


def apply_patch_mrope():
    import torch
    from megatron.core.models.common.embeddings import MultimodalRotaryEmbedding

    def patch_multimodal_rotary_embedding_forward(self, position_ids: torch.Tensor, mrope_section):
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
        else:
            bs = freqs.shape[1]
            emb = torch.stack((freqs.view(3, bs, -1, 1), freqs.view(3, bs, -1, 1)), dim=-1).view(
                3, bs, freqs.shape[0], -1
            )

        # generate freqs with mrope_section
        # shape (bs, seq_length, 2 * dim)
        mrope_section = mrope_section * 2
        emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()

        # mrope in mcore 0.12 can only support sbhd format context parallel, but qwen2_5_vl in verl use thd format.
        # we need to disable it here.
        # if parallel_state.get_context_parallel_world_size() > 1:
        #     # slice rotary_pos_emb along sequence dimension and select the parition of the current
        #     # CP rank
        #     emb = get_pos_emb_on_this_cp_rank(emb, 1)
        return emb

    MultimodalRotaryEmbedding.forward = patch_multimodal_rotary_embedding_forward
