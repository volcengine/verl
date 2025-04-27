# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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
from typing import List
from dataclasses import dataclass, field
from megatron.core.transformer import TransformerConfig
from megatron.training.activations import quick_gelu
from megatron.core import parallel_state

@dataclass
class Qwen2VLTransformerConfig(TransformerConfig):

    transformer_impl: str = 'transformer_engine'
    rotary_base: int = None
    rotary_scaling_factor: int = None
    max_position_embeddings: int = None
    moe_aux_loss_coeff: float = 0.0
    mrope_section: List[int] = field(default_factory=lambda:[16, 24, 24])

    # The following options are set with --disable-bias-linear --add-qkv-bias
    # in the script
    # add_bias_linear = False
    # add_qkv_bias = True
    


def get_vision_model_config(args, config):
    # mlp: embed_dim -> embed_dim * mlp_ratio -> embed_dim, silu
    # NOTE: here we provide a workaround to solve the wrong layer amount when VPP of decoder is on
    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        config.num_layers = 32 * parallel_state.get_virtual_pipeline_model_parallel_world_size() # depth
    else:
        config.num_layers = 32 # depth
    config.num_attention_heads = 16 # num_heads
    config.add_bias_linear = True # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True # qkv_proj in attn has bias
    config.hidden_size = 1280 # embed_dim
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 1280 * 4 # embed_dim * mlp_ratio
    config.gated_linear_unit = False # no gated
    config.activation_func = quick_gelu # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads # no GQA
    config.layernorm_zero_centered_gamma = False # False
    config.apply_query_key_layer_scaling = False # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False # no swiglu, set false
    config.bias_dropout_fusion = False # no dropout, set false
    config.attention_softmax_in_fp32 = True # use True
    config.normalization = 'LayerNorm' # use LayerNorm
    config.seq_length = args.seq_length

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = 14
    config.in_channels = 3
    config.spatial_merge_size = 2
    return config


def get_vision_projection_config(config, embed_dim, spatial_merge_size):
    # merger: 
    # context_dim = embed_dim * merge_size**2
    # context_dim -> context_dim -> hidden_size
    # MLP: 
    # input_size -> ffn_hidden_size -> hidden_size
    # spec: LN -> Linear(bias=True) -> GELU -> Linear(bias=True)
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = embed_dim * (spatial_merge_size ** 2)
    config.activation_func = torch.nn.functional.gelu
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    return config
