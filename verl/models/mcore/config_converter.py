# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from transformers import PretrainedConfig


def _get_base_transformer_config(hf_config: PretrainedConfig, dtype: torch.dtype, **kwargs) -> TransformerConfig:
    """
    Create a base TransformerConfig with common parameters across different model architectures.
    TODO: (ycl) use dataclass or converter config?

    Args:
        hf_config: HuggingFace model configuration
        dtype: Data type for the model
        **kwargs: Additional parameters to override defaults

    Returns:
        TransformerConfig with common parameters
    """
    from megatron.core import parallel_state as mpu

    # Common parallel state parameters
    overlap_p2p_comm = mpu.get_virtual_pipeline_model_parallel_world_size() is not None and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    batch_p2p_comm = False

    # Base configuration with common parameters
    base_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "layernorm_epsilon": hf_config.rms_norm_eps,
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel configuration
        "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": mpu.get_context_parallel_world_size(),
        "overlap_p2p_comm": overlap_p2p_comm,
        "batch_p2p_comm": batch_p2p_comm,
        "sequence_parallel": mpu.get_tensor_model_parallel_world_size() > 1,
        # Common settings
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
    }

    # Update with any provided overrides
    base_config.update(kwargs)
    print(f"Overridden TF init config: {base_config}")

    return TransformerConfig(**base_config)


def hf_to_mcore_config_dense(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # for LlamaForCausalLM or Qwen2ForCausalLM
    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    qk_layernorm = True if "Qwen3ForCausalLM" in hf_config.architectures else False

    return _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=qkv_bias,
        qk_layernorm=qk_layernorm,
    )


def hf_to_mcore_config_qwen2moe(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    return _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="aux_loss",
        moe_shared_expert_overlap=True,
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # Other optimizations
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Qwen specific
        moe_router_pre_softmax=True,
        add_qkv_bias=True,
    )


def hf_to_mcore_config_mixtral(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    return _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        num_moe_experts=hf_config.num_local_experts,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        moe_router_topk=hf_config.num_experts_per_tok,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="aux_loss",
        moe_router_score_function="softmax",
        moe_shared_expert_intermediate_size=None,  # mixtral has no shared expert
        moe_shared_expert_overlap=False,  # mixtral has no shared expert
        moe_ffn_hidden_size=hf_config.intermediate_size,
        moe_router_bias_update_rate=0.001,
        # moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        # Other optimizations
        persist_layer_norm=True,
        apply_rope_fusion=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
    )


def hf_to_mcore_config_qwen3moe(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    return _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="aux_loss",
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # Other optimizations
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Qwen specific
        moe_router_pre_softmax=True,
        qk_layernorm=True,
    )


def hf_to_mcore_config_dpskv3(hf_config: PretrainedConfig, dtype: torch.dtype) -> MLATransformerConfig:
    # DeepseekV3ForCausalLM
    raise NotImplementedError("DeepseekV3ForCausalLM is not supported yet")


def hf_to_mcore_config_qwen2_5_vl(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Qwen2_5_VLForConditionalGeneration
    raise NotImplementedError("Qwen2_5_VLForConditionalGeneration is not supported yet")


def hf_to_mcore_config_llama4(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Llama4ForConditionalGeneration
    raise NotImplementedError("Llama4ForConditionalGeneration is not supported yet")
