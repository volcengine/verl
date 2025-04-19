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

import os
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_npu
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision, \
    Qwen2_5_VLVisionFlashAttention2, Qwen2_5_VLVisionSdpaAttention
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.integrations import npu_flash_attention
from transformers.utils import logging

TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", default=DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE))
if SPARSE_MODE not in [TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE, DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE]:
    raise ValueError(
        "Environment variable `NPU_FA2_SPARSE_MODE` can only be set as 2 (top-left aligned causal mask) "
        "or 3 (down-right aligned causal mask)."
    )

logger = logging.get_logger(__name__)

def npu_flash_attn_func_patch(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if not causal:
        head_num = q.shape[2]
        output = torch_npu.npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=keep_prob, scale=softmax_scale)[0]
    else:
        attn_mask_npu = torch.triu(torch.ones([2048, 2048], device=q.device), diagonal=1).bool()
        head_num = q.shape[2]
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            "BSND",
            keep_prob=keep_prob,
            scale=softmax_scale,
            atten_mask=attn_mask_npu,
            sparse_mode=SPARSE_MODE,
        )[0]

    return output


def npu_flash_attn_varlen_func_patch(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,  # defined for aligning params order with corresponding function in `flash-attn`
    max_seqlen_k=None,  # defined for aligning params order with corresponding function in `flash-attn`
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if not causal:
        head_num = q.shape[1]
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=None,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
        )[0]
    else:
        attn_mask_npu = torch.triu(torch.ones([2048, 2048], device=q.device), diagonal=1).bool()
        head_num = q.shape[1]
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask_npu,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            sparse_mode=SPARSE_MODE,
        )[0]

    return output

def apply_rotary_pos_emb_flashatt_npu(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = apply_rotary_emb(q.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()).type_as(k)
    return q_embed, k_embed


def vision_fa2_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_flashatt_npu(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    attn_output = npu_flash_attn_varlen_func_patch(q, k, v, cu_seqlens, cu_seqlens).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output

def sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask.unsqueeze(0), dropout_p=0.0
    )
    attn_output = attn_output.squeeze(0).transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = vision_fa2_forward
Qwen2_5_VLVisionSdpaAttention.forward = sdpa_forward
modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt_npu
npu_flash_attention.npu_flash_attn_varlen_func = npu_flash_attn_varlen_func_patch
npu_flash_attention.npu_flash_attn_func = npu_flash_attn_func_patch
