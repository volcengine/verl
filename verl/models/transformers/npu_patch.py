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
import transformers
from transformers.models import (
    llama,
    qwen2,
    qwen2_5_vl,
    qwen2_moe,
    qwen3,
    qwen3_moe,
)

from verl.utils.device import get_device_name

# Define model classes mapping
MODEL_CLASSES = {
    "llama": [
        llama.modeling_llama.LlamaMLP,
        llama.modeling_llama.LlamaRMSNorm,
    ],
    "qwen2": [
        qwen2.modeling_qwen2.Qwen2MLP,
        qwen2.modeling_qwen2.Qwen2RMSNorm,
    ],
    "qwen2_moe": [
        qwen2_moe.modeling_qwen2_moe.Qwen2MoeMLP,
        qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm,
    ],
    "qwen2_5_vl": [
        qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLMLP,
        qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm,
        qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt,
    ],
    "qwen3": [
        qwen3.modeling_qwen3.Qwen3MLP,
        qwen3.modeling_qwen3.Qwen3RMSNorm,
        qwen3.modeling_qwen3.apply_rotary_pos_emb,
    ],
    "qwen3_moe": [
        qwen3_moe.modeling_qwen3_moe.Qwen3MoeMLP,
        qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm,
        qwen3_moe.modeling_qwen3_moe.apply_rotary_pos_emb,
    ],
}


def npu_rotary_pos_emb_vl(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NPU optimized rotary position embedding for vlm model"""
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    cos = cos.repeat(1, 2)
    sin = sin.repeat(1, 2)
    q_embed = torch_npu.npu_rotary_mul(
        q.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(q)
    k_embed = torch_npu.npu_rotary_mul(
        k.float(), cos.unsqueeze(0).unsqueeze(2).float(), sin.unsqueeze(0).unsqueeze(2).float()
    ).type_as(k)
    return q_embed, k_embed


def npu_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """NPU optimized rotary position embedding"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def npu_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
    """NPU optimized RMSNorm"""
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def npu_silu(self, hidden_state: torch.Tensor) -> torch.Tensor:
    """NPU optimized silu"""
    gate_up = torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))


def apply_patches():
    """
    NOTE1: Only works on transformers==4.52.4.
    NOTE2: These patches are temporary.
    After these fusion kernels are integrated in transformers, we will remove the all npu_patch.
    For tracking integration progress: https://github.com/huggingface/transformers/issues/39105.
    """

    apply_rotary_pos_emb_vlm_models = ["qwen2_5_vl"]
    apply_rotary_pos_emb_models = ["qwen3", "qwen3_moe"]
    apply_rms_norm_and_silu_models = ["llama", "qwen2", "qwen2_moe", "qwen2_5_vl", "qwen3", "qwen3_moe"]

    # Patch rotary embedding for supported vlm models
    for model in apply_rotary_pos_emb_vlm_models:
        if model in MODEL_CLASSES:
            MODEL_CLASSES[model][2] = npu_rotary_pos_emb_vl

    # Patch rotary embedding for supported models
    for model in apply_rotary_pos_emb_models:
        if model in MODEL_CLASSES:
            MODEL_CLASSES[model][2] = npu_rotary_pos_emb

    # Patch RMSNorm and silu for supported models
    for model in apply_rms_norm_and_silu_models:
        if model in MODEL_CLASSES:
            MODEL_CLASSES[model][1].forward = npu_rms_norm
            MODEL_CLASSES[model][0].forward = npu_silu


# Apply all patches
if get_device_name() == "npu":
    if transformers.__version__ != "4.52.4":
        raise ImportError("NPU fusion kernels patch only works with transformers==4.52.4")
    else:
        apply_patches()
        print(
            "Applied some optimized fusion kernels patch on npu. "
            "Check verl.models.transformers.npu_patch.py for more details."
        )
