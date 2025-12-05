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


from importlib.metadata import version as get_version
from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.activations import ACT2FN
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
from transformers.utils import logging

if get_version("transformers") > "4.57.1":
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel
else:
    from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

logger = logging.get_logger(__name__)


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
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def silu_forward(self, hidden_state):
    """NPU optimized silu"""
    gate_up = torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list, group_list_type=1):
        """
        Grouped Matmul(GMM) for Ascend NPU.

        Args:
            x (torch.Tensor): Input tensor, shape (tokens_num * top_k, hidden_size)
            weight (torch.Tensor): Expert weights, shape (n_experts, hidden_size, intermediate_size)
            group_list (torch.Tensor): Expert token counts, shape (n_experts,)
                - type 0: cumsum of tokens per expert
                - type 1: direct tokens per expert (default)
        """
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.group_list_type = group_list_type

        output = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=group_list_type
        )[0]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        group_list_type = ctx.group_list_type

        dx = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]

        dw = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )[0]

        return dx, dw, None, None


def moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states: (batch_size, sequence_length, hidden_size)
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Loop over all available experts in the model and perform the computation on each expert
    # Concat all weights
    input_dtype = hidden_states.dtype
    up_weight_list = [e.up_proj.weight for e in self.experts]
    gate_weight_list = [e.gate_proj.weight for e in self.experts]
    down_weight_list = [e.down_proj.weight for e in self.experts]
    w1 = torch.stack(up_weight_list).transpose(1, 2).to(input_dtype)
    w2 = torch.stack(gate_weight_list).transpose(1, 2).to(input_dtype)
    w3 = torch.stack(down_weight_list).transpose(1, 2).to(input_dtype)

    permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)

    up_res = GmmFunction.apply(permuted_tokens, w1, tokens_per_expert)
    gate_res = GmmFunction.apply(permuted_tokens, w2, tokens_per_expert)
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
    down_res = GmmFunction.apply(act_res, w3, tokens_per_expert)

    final_hidden_states = torch_npu.npu_moe_token_unpermute(down_res, row_ids_map, probs=routing_weights)

    return final_hidden_states, router_logits


class NPUQwen3VLMoeTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once.
        By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        if self.training:
            permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
                hidden_states, router_indices.to(torch.int32)
            )
            tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
            intermediate_hidden_states = GmmFunction.apply(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
            output = GmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)
            next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = NPUQwen3VLMoeTextExperts(config)

        # since all the models use norm_topk_prob, we don't need to have a extra check for it
        # self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        if not self.training:
            routing_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        routed_out = self.experts(hidden_states, routing_weights, router_indices)
        return routed_out


@classmethod
def _check_and_enable_flash_attn_2(
    cls,
    config,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str | dict[str, int]] = None,
    check_device_map: bool = True,
    hard_check_only: bool = False,
) -> PretrainedConfig:
    """
    Checks the availability of Flash Attention 2 and compatibility with the current model.

    If all checks pass and `hard_check_only` is False, the method will set the config attribute
    `attn_implementation` to "flash_attention_2" so that the model can initialize
    the correct attention module.
    """
    if not cls._supports_flash_attn_2:
        raise ValueError(
            f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where the"
            f" model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
            " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
        )

    if not hard_check_only:
        config._attn_implementation = "flash_attention_2"
    logger.info("Detect using FlashAttention2 on Ascend NPU.")
    return config


modeling_qwen2.Qwen2RMSNorm.forward = rms_norm_forward
modeling_qwen2.Qwen2MLP.forward = silu_forward
modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_npu
modeling_qwen2_5_vl.Qwen2RMSNorm.forward = rms_norm_forward
modeling_qwen2_5_vl.Qwen2_5_VLMLP.forward = silu_forward
modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = apply_rotary_pos_emb_flashatt_npu
modeling_qwen3_moe.Qwen3MoeRMSNorm.forward = rms_norm_forward
modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = moe_block_forward
modeling_qwen3_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_npu
modeling_qwen3.Qwen3RMSNorm.forward = rms_norm_forward
modeling_qwen3.Qwen3MLP.forward = silu_forward
modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = Qwen3VLMoeTextSparseMoeBlock
modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm.forward = rms_norm_forward
modeling_qwen3_vl_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_npu
modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = rms_norm_forward
modeling_qwen3_vl.Qwen3VLTextMLP.forward = silu_forward

if get_version("transformers") < "4.54.0":
    PreTrainedModel._check_and_enable_flash_attn_2 = _check_and_enable_flash_attn_2
