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

from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config
from megatron.core import ModelParallelConfig

from .parallel_attention import ParallelQwen2Attention, ParallelQwen2AttentionRmPad
from .parallel_mlp import ParallelQwen2MLP
from .parallel_rmsnorm import ParallelQwen2RMSNorm


class ParallelQwen2DecoderLayer(nn.Module):

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelQwen2Attention(config=config, megatron_config=megatron_config)

        self.mlp = ParallelQwen2MLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelQwen2RMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Note: sequence parallel is hidden inside ColumnParallelLinear
        # reduce scatter is hidden inside RowParallelLinear

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # TODO: add sequence parallel operator reduce_scatter here

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # TODO: add sequence parallel operator all_gather here

        hidden_states = self.mlp(hidden_states)

        # TODO: add sequence parallel operator reduce_scatter here

        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class ParallelQwen2DecoderLayerRmPad(nn.Module):

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelQwen2AttentionRmPad(config=config, megatron_config=megatron_config)

        self.mlp = ParallelQwen2MLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelQwen2RMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states  # (total_nnz // sp, 1, hidden_size)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # (total_nnz // sp, 1, hidden_size) -> all-gather (total_nnz, 1, hidden_size)
        # -> col + row -> reduce-scatter -> (total_nnz // sp, 1, hidden_size)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       position_ids=position_ids,
                                       sequence_length=sequence_length,
                                       indices=indices,
                                       cu_seqlens=cu_seqlens,
                                       max_seqlen_in_batch=max_seqlen_in_batch)

        hidden_states = residual + hidden_states

        # Fully Connected
        # shape changes same as attn
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs
