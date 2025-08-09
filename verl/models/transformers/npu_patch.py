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


from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_npu
import transformers
from torch import nn
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.activations import ACT2FN
from transformers.cache_utils import (Cache, DynamicCache, SlidingWindowCache,
                                      StaticCache)
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           MoeCausalLMOutputWithPast,
                                           MoeModelOutputWithPast,
                                           QuestionAnsweringModelOutput,
                                           SequenceClassifierOutputWithPast,
                                           TokenClassifierOutput)
from transformers.modeling_rope_utils import (ROPE_INIT_FUNCTIONS,
                                              dynamic_rope_update)
from transformers.modeling_utils import (ALL_ATTENTION_FUNCTIONS,
                                         PreTrainedModel)
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from transformers.processing_utils import Unpack
from transformers.utils import (LossKwargs, auto_docstring, can_return_tuple,
                                is_torch_flex_attn_available, logging)


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


def apply_rotary_pos_emb_qwen3_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class torch_npu_gmm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group_list, cpy_stream):
        outputs = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, group_type=0, split_item=2)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        with torch_npu.npu.stream(cpy_stream):
            ctx.cpu_group_list = group_list.to('cpu', non_blocking=True)

        ctx.cpy_stream = cpy_stream
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        wt = weight.permute(0, 2, 1)
        xt = x.permute(1, 0)
        dx = torch_npu.npu_grouped_matmul([grad_outputs], [wt], group_list=group_list, group_type=0, split_item=2)
        dw = torch.zeros_like(weight)
        cpy_stream = ctx.cpy_stream
        cpy_stream.synchronize()
        cpu_group_list = [0] + ctx.cpu_group_list.tolist()
        for i in range(weight.shape[0]):
            if (cpu_group_list[i+1] - cpu_group_list[i]) != 0:
                dw[i] = torch.matmul(xt[:,cpu_group_list[i]:cpu_group_list[i+1]], grad_outputs[cpu_group_list[i]:cpu_group_list[i+1]])

        return dx[0], dw, None, None


class Qwen3MoeSparseMoeBlockPatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        # Concat all weights
        input_dtype = hidden_states.dtype
        up_weight_list = [e.up_proj.weight.t().to(input_dtype) for e in self.experts]
        gate_weight_list = [e.gate_proj.weight.t().to(input_dtype) for e in self.experts]
        down_weight_list = [e.down_proj.weight.t().to(input_dtype) for e in self.experts]
        w1 = torch.stack(up_weight_list)
        w2 = torch.stack(gate_weight_list)
        w3 = torch.stack(down_weight_list)

        # Copied from mindspeed moe_utils.py:permute
        routing_map = selected_experts
        flatten_indices = routing_map.view(-1)
        sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
        permuted_tokens = hidden_states.index_select(0, sorted_indices // self.top_k)

        tokens_per_experts = torch.sum(expert_mask, dim=(1,2))
        group_list = torch.cumsum(tokens_per_experts, dim=0)

        cpy_stream = torch_npu.npu.Stream()
        up_res = torch_npu_gmm.apply(permuted_tokens, w1, group_list, cpy_stream)
        gate_res = torch_npu_gmm.apply(permuted_tokens, w2, group_list, cpy_stream)
        act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
        down_res = torch_npu_gmm.apply(act_res, w3, group_list, cpy_stream)
       
        probs = routing_weights
        num_unpermuted_tokens = probs.numel()
        topk = self.top_k
        permuted_tokens = down_res

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1).to(hidden_states.dtype)
        final_hidden_states = unpermuted_tokens

        return final_hidden_states, router_logits


Qwen2RMSNorm.forward = rms_norm_forward
modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt_npu
transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm.forward = rms_norm_forward
transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlockPatch
transformers.models.qwen3_moe.modeling_qwen3_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_qwen3_npu
