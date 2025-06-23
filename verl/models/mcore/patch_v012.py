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


from collections import OrderedDict
from typing import Optional

import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from torch import Tensor

from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.model import CausalLMOutputForPPO


class GPTModelWithFusedLayer(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> CausalLMOutputForPPO:
        """
        Forward pass for GPT models with fused kernel support.

        Patch https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == "rope" and not self.config.multi_latent_attention:
            if not self.training and self.config.flash_decode and inference_context:
                assert inference_context.is_static_batching(), "GPTModel currently only supports static inference batching."
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(inference_context, self.decoder, decoder_input, self.config, packed_seq_params)
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == "thd",
                )
        elif self.position_embedding_type == "mrope" and not self.config.multi_latent_attention:
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError("Flash decoding uses precomputed cos and sin for RoPE, not implmented in MultimodalRotaryEmbedding yet.")

        if (self.config.enable_cuda_graph or self.config.flash_decode) and rotary_pos_cos is not None and inference_context and inference_context.is_static_batching() and not self.training:
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * inference_context.current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # skip inference

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        # Process inference output.
        if inference_context and not inference_context.is_static_batching():
            hidden_states = inference_context.last_token_logits(hidden_states.squeeze(1).unsqueeze(0)).unsqueeze(1)

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                output_layer=self.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                compute_language_model_loss=self.compute_language_model_loss,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        output = CausalLMOutputForPPO(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )

        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        logprobs, entropy = linear_cross_entropy(
            hidden_states,
            self.output_layer.weight,
            labels,
            temperature,
            "none",
            parallel_state.get_tensor_model_parallel_group(),
        )

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "decoder_input": decoder_input,
                    "logprobs": logprobs,
                    "entropy": entropy,
                }
            )
            log_config_to_disk(self.config, payload, prefix="input_and_logits")

        output.entropy = entropy
        output.log_probs = logprobs

        return output


def apply_patch_dpskv3():
    import torch
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention, apply_rotary_pos_emb, deprecate_inference_params, gather_from_sequence_parallel_region, gather_from_tensor_model_parallel_region, scatter_to_sequence_parallel_region

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
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(inference_context, None, hidden_states, self.config, packed_seq_params)

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
            kv_compressed, k_pos_emb = torch.split(kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
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
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value

    MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors


def apply_monkey_patch(use_dpskv3_patch: bool = False):
    """_summary_
    Patch for deepseek-V3 in mcore 0.12.

    Args:
        use_dpskv3_patch (bool, optional): Use patch for deepseek-V3. Defaults to False.
    """
    if use_dpskv3_patch:
        apply_patch_dpskv3()
