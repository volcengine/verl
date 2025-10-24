# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.layers.rotary import apply_rotary_emb
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn
from transformers import LlamaConfig
from transformers.utils import is_flash_attn_2_available

from verl.models.llama.megatron.layers.parallel_linear import QKVParallelLinear
from verl.utils.megatron import tensor_parallel as tp_utils


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaLlama3ScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, dim, config, max_position_embeddings=2048, base=10000, device=None):
        super().__init__(dim, max_position_embeddings, base, device)

        self.factor = config.rope_scaling["factor"]  # `8` in the original implementation
        self.high_freq_factor = config.rope_scaling["high_freq_factor"]  # `1` in the original implementation
        self.low_freq_factor = config.rope_scaling["low_freq_factor"]  # `4` in the original implementation
        self.old_context_len = config.rope_scaling[
            "original_max_position_embeddings"
        ]  # `8192` in the original implementation

        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor

        wavelen = 2 * math.pi / self.inv_freq
        # wavelen < high_freq_wavelen: do nothing; wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / self.factor, self.inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / self.factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ParallelLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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

        # assign values after tp
        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0, (
            f"num_head must be divisible by tp_size. Got num_head={self.num_heads}, tp_size={tp_size}"
        )
        assert self.num_key_value_heads % tp_size == 0, (
            f"num_key_value_heads must be divisible by tp_size. Got num_key_value_heads="
            f"{self.num_key_value_heads}, tp_size={tp_size}"
        )

        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_key_value_heads_per_tp = self.num_key_value_heads // tp_size
        self.hidden_size_per_tp = self.hidden_size // tp_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            assert row_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # [self.q_size, self.k_size, self.v_size]
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

        self.q_size = self.num_heads_per_tp * self.head_dim
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim

        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            rope_type_key = "type" if "type" in self.config.rope_scaling else "rope_type"
            scaling_type = self.config.rope_scaling[rope_type_key]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "llama3":
                self.rotary_emb = LlamaLlama3ScalingRotaryEmbedding(
                    self.head_dim,
                    self.config,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)[0]
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads_per_tp, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads_per_tp, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads_per_tp, q_len, kv_seq_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads_per_tp, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads_per_tp, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_tp)
        attn_output = self.o_proj(attn_output)[0]
        return attn_output


"""
Remove padding Attention
- Using Flash-attn 2
- Compatible with sequence parallel
"""


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


def apply_rotary_pos_emb_rmpad(q, k, cos, sin, position_ids, indices, sequence_length):
    batch_size = position_ids.shape[0]

    q = pad_input(q, indices, batch_size, sequence_length)  # (batch_size, seqlen, num_head, head_dim)
    k = pad_input(k, indices, batch_size, sequence_length)
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = index_first_axis(rearrange(q_embed, "b s ... -> (b s) ..."), indices)
    k_embed = index_first_axis(rearrange(k_embed, "b s ... -> (b s) ..."), indices)

    return q_embed, k_embed


# use flash-attn rotary embeddings with rmpad
# cos/sin shoudl be: (seq_length, rotary_dim / 2)
def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
    q_embed = apply_rotary_emb(
        q, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )
    k_embed = apply_rotary_emb(
        k, cos, sin, interleaved=False, inplace=False, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
    )
    return q_embed, k_embed


class ParallelLlamaAttentionRmPad(ParallelLlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen_in_batch: int = None,
    ):
        total_nnz, _, _ = hidden_states.size()  # This is the total_nnz padded after sequence parallel

        if self.megatron_config.sequence_parallel:
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

        qkv = self.qkv_proj(hidden_states)[0]
        query_states, key_states, value_states = qkv.split(
            [self.q_size, self.k_size, self.v_size], dim=-1
        )  # (total_nnz, 1, hidden_size)

        if self.megatron_config.sequence_parallel:
            sequence_parallel_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]  # total_nnz before sp padding
            query_states = query_states[:total_nnz]
            key_states = key_states[:total_nnz]
            value_states = value_states[:total_nnz]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(total_nnz, self.num_heads_per_tp, self.head_dim)
        key_states = key_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)
        value_states = value_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)

        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        cos, sin = cos[:, : cos.shape[1] // 2], sin[:, : sin.shape[1] // 2]  # flash attn only needs half
        query_states, key_states = apply_rotary_pos_emb_rmpad_flash(
            query_states, key_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch
        )
        # query_states, key_states = apply_rotary_pos_emb_rmpad(query_states, key_states, cos, sin,
        # position_ids, indices,

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=True,
        )

        attn_output_unpad = attn_output_unpad.to(input_dtype)
        attn_output_unpad = attn_output_unpad.reshape(total_nnz, 1, self.hidden_size_per_tp).contiguous()

        # sequence parallel reduce_scatter is performed inside RowColumnParallel if enabled
        # Here we need to repad
        if self.megatron_config.sequence_parallel:
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        attn_output_unpad = self.o_proj(attn_output_unpad)[0]
        return attn_output_unpad
