# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision, \
    Qwen2_5_VLVisionSdpaAttention
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.utils import logging


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb_flashatt_npu(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = apply_rotary_emb(q.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()).type_as(k)
    return q_embed, k_embed


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
    # scaled_dot_product_attention on ASCEND NPU only supports the case where q/k/v/attention_mask is both
    # 4-dimensional, so attention_mask can be changed to 4-dimensional through unsqueeze
    attn_output = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask.unsqueeze(0), dropout_p=0.0
    )
    attn_output = attn_output.squeeze(0).transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


Qwen2_5_VLVisionSdpaAttention.forward = sdpa_forward
modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt_npu
