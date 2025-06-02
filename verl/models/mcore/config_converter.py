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


def _get_base_transformer_config(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> dict:
    """
    Create a base TransformerConfig with common parameters across different model architectures.
    TODO: (ycl) use dataclass or converter config?

    Args:
        hf_config: HuggingFace model configuration
        dtype: Data type for the model
        override_transformer_config_kwargs: Additional parameters to override defaults

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
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
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
    base_config.update(override_transformer_config_kwargs)
    print(f"Overridden TF init config: {base_config}")

    return base_config


def hf_to_mcore_config_dense(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> TransformerConfig:
    # for LlamaForCausalLM or Qwen2ForCausalLM
    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    qk_layernorm = True if "Qwen3ForCausalLM" in hf_config.architectures else False

    args = _get_base_transformer_config(hf_config=hf_config, dtype=dtype, use_cpu_initialization=False, add_bias_linear=False, add_qkv_bias=qkv_bias, qk_layernorm=qk_layernorm, **override_transformer_config_kwargs)
    return TransformerConfig(**args)


def hf_to_mcore_config_qwen2moe(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> TransformerConfig:
    args = _get_base_transformer_config(
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
        **override_transformer_config_kwargs,
    )
    return TransformerConfig(**args)


def hf_to_mcore_config_mixtral(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> TransformerConfig:
    args = _get_base_transformer_config(
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
        **override_transformer_config_kwargs,
    )
    return TransformerConfig(**args)


def hf_to_mcore_config_qwen3moe(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> TransformerConfig:
    args = _get_base_transformer_config(
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
        moe_router_pre_softmax=False,
        qk_layernorm=True,
        **override_transformer_config_kwargs,
    )
    return TransformerConfig(**args)


def hf_to_mcore_config_dpskv3(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> MLATransformerConfig:
    # DeepseekV3ForCausalLM
    from megatron.core.transformer.enums import AttnBackend

    from .patch_v012 import apply_patch

    apply_patch()

    mla_rope_config = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "rope",
    }
    if "rope_scaling" in hf_config and hf_config.rope_scaling is not None:
        mla_rope_config.update(hf_config.rope_scaling)
    moe_layer_freq = [1] * hf_config.num_hidden_layers
    for i in range(hf_config.first_k_dense_replace):
        moe_layer_freq[i] = 0

    args = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        attention_backend=AttnBackend.fused,
        bf16=dtype is torch.bfloat16,
        layernorm_epsilon=hf_config.rms_norm_eps,
        ffn_hidden_size=hf_config.intermediate_size,
        qk_layernorm=True,
        # moe specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_bias_update_rate=0.001,
        moe_router_enable_expert_bias=True,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.n_routed_experts,
        moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
        moe_aux_loss_coeff=getattr(hf_config, "aux_loss_alpha", 0.001),
        moe_router_load_balancing_type="seq_aux_loss",
        moe_shared_expert_overlap=True,
        # moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_layer_freq=moe_layer_freq,
        # MLA
        q_lora_rank=hf_config.q_lora_rank,
        kv_lora_rank=hf_config.kv_lora_rank,
        qk_head_dim=hf_config.qk_nope_head_dim,
        qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
        v_head_dim=hf_config.v_head_dim,
        rotary_base=hf_config.rope_theta,
        rotary_scaling_factor=mla_rope_config["factor"],
        rope_type=mla_rope_config["type"],
        mscale=mla_rope_config["mscale"],
        mscale_all_dim=mla_rope_config["mscale_all_dim"],
        max_position_embeddings=mla_rope_config["original_max_position_embeddings"],
        beta_fast=mla_rope_config["beta_fast"],
        beta_slow=mla_rope_config["beta_slow"],
        # mcore 0.12 moe
        moe_router_dtype="fp64",
        disable_bf16_reduced_precision_matmul=True,
        # other
        # deallocate_pipeline_outputs=True,
        # gradient_accumulation_fusion=True,
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        **override_transformer_config_kwargs,
    )
    transformer_config = MLATransformerConfig(**args)

    return transformer_config


def hf_to_mcore_config_qwen2_5_vl(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> TransformerConfig:
    # Qwen2_5_VLForConditionalGeneration
    raise NotImplementedError("Qwen2_5_VLForConditionalGeneration is not supported yet")


def hf_to_mcore_config_llama4(hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs) -> TransformerConfig:
    # Llama4ForConditionalGeneration
    raise NotImplementedError("Llama4ForConditionalGeneration is not supported yet")
