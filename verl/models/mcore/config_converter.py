# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# convert huggingface config to mcore transformer config

import torch
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from transformers import PretrainedConfig


def hf_to_mcore_config_dense(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # for LlamaForCausalLM or Qwen2ForCausalLM
    from megatron.core import parallel_state as mpu

    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    overlap_p2p_comm = (
        mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    batch_p2p_comm = False
    transformer_config = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        activation_func=F.silu,
        normalization="RMSNorm",
        gated_linear_unit=True,
        use_cpu_initialization=True,
        add_bias_linear=False,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        pipeline_dtype=dtype,
        params_dtype=dtype,
        sequence_parallel=mpu.get_tensor_model_parallel_world_size() > 1,
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        moe_token_dispatcher_type="alltoall",
        attention_dropout=hf_config.attention_dropout,
        hidden_dropout=getattr(hf_config, "hidden_dropout", 0.0),
        add_qkv_bias=qkv_bias,
        attention_backend=AttnBackend.flash,
        bf16=dtype is torch.bfloat16,
    )

    return transformer_config


def hf_to_mcore_config_qwen2moe(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Qwen2MoeForCausalLM
    raise NotImplementedError("Qwen2MoeForCausalLM is not supported yet")


def hf_to_mcore_config_dpskv3(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # DeepseekV3ForCausalLM
    raise NotImplementedError("DeepseekV3ForCausalLM is not supported yet")


def hf_to_mcore_config_qwen2_5_vl(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Qwen2_5_VLForConditionalGeneration
    raise NotImplementedError("Qwen2_5_VLForConditionalGeneration is not supported yet")


def hf_to_mcore_config_llama4(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Llama4ForConditionalGeneration
    raise NotImplementedError("Llama4ForConditionalGeneration is not supported yet")
