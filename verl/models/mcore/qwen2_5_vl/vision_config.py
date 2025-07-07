# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
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
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig


def get_vision_model_config(config: TransformerConfig) -> TransformerConfig:
    # Given a Transformer Config from decoder, build vision encoder config
    # diff: out_hidden_size & intermediate_size

    # mlp: hidden_size -> intermediate_size -> embed_dim, silu
    # NOTE: here we provide a workaround to solve the wrong layer amount when VPP of decoder is on
    if config.num_layers in [28, 36]:
        config.ffn_hidden_size = 3420
    else:
        config.ffn_hidden_size = 3456

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        config.num_layers = 32 * parallel_state.get_virtual_pipeline_model_parallel_world_size()  # depth
    else:
        config.num_layers = 32  # depth
    config.num_attention_heads = 16  # num_heads
    config.add_bias_linear = True  # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True  # qkv_proj in attn has bias
    config.hidden_size = 1280  # hidden_size
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    # config.gated_linear_unit = False # no gated
    # config.activation_func = quick_gelu # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  # no GQA
    config.layernorm_zero_centered_gamma = False  # False
    config.apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False  # no swiglu, set false
    config.bias_dropout_fusion = False  # no dropout, set false
    config.attention_softmax_in_fp32 = True  # use True
    # config.normalization = 'LayerNorm' # use RMSNorm
    config.seq_length = 1

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = 14
    config.in_channels = 3
    config.spatial_merge_size = 2

    config.fullatt_block_indexes = [7, 15, 23, 31]
    config._qwen2_5_vl_window_size = 112
    return config


def get_vision_projection_config(
    config: TransformerConfig, embed_dim: int, spatial_merge_size: int
) -> TransformerConfig:
    # merger:
    # context_dim = hidden_size * merge_size**2
    # out_hidden_size = hidden_size
    # context_dim -> context_dim -> out_hidden_size
    # MLP:
    # input_size -> ffn_hidden_size -> hidden_size
    # spec: LN -> Linear(bias=True) -> GELU -> Linear(bias=True)
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = embed_dim * (spatial_merge_size**2)
    config.activation_func = torch.nn.functional.gelu
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    return config
