# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
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


import torch
import torch_npu
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm


# This patch takes effect when using apply_rotary_pos_emb_flashatt on qwen2_5_vl and will be removed in
# subsequent versions
# https://github.com/huggingface/transformers/pull/38491
def apply_rotary_pos_emb_flashatt_npu(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    cos = cos.repeat(1, 2)
    sin = sin.repeat(1, 2)
    q_embed = apply_rotary_emb(
        q.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(q)
    k_embed = apply_rotary_emb(
        k.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(k)
    return q_embed, k_embed


# This api can improve performance on ASCEND NPU
def rms_norm_forward(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


Qwen2RMSNorm.forward = rms_norm_forward
modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt_npu
