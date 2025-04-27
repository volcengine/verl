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
from megatron.core import parallel_state as mpu
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
    from megatron.core import parallel_state as mpu

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
        attention_dropout=hf_config.attention_dropout,
        hidden_dropout=getattr(hf_config, "hidden_dropout", 0.0),
        activation_func=F.silu,
        normalization="RMSNorm",
        gated_linear_unit=True,
        use_cpu_initialization=False,
        add_bias_linear=False,
        pipeline_dtype=dtype,
        params_dtype=dtype,
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        attention_backend=AttnBackend.flash,
        # attention_backend=AttnBackend.fused,
        bf16=dtype is torch.bfloat16,
        layernorm_epsilon=hf_config.rms_norm_eps,
        ffn_hidden_size=hf_config.intermediate_size,
        # parallel config
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        sequence_parallel=mpu.get_tensor_model_parallel_world_size() > 1,
        # moe specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="aux_loss",
        moe_shared_expert_overlap=True,
        # moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # # mcore 0.12 moe
        # moe_router_dtype="fp64",
        # disable_bf16_reduced_precision_matmul=True,
        # other
        # deallocate_pipeline_outputs=True,
        # gradient_accumulation_fusion=True,
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # qwen specific
        moe_router_pre_softmax=True,
        add_qkv_bias=True,
    )
    return transformer_config


def hf_to_mcore_config_dpskv3(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # DeepseekV3ForCausalLM
    raise NotImplementedError("DeepseekV3ForCausalLM is not supported yet")


def hf_to_mcore_config_qwen2_5_vl(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Qwen2_5_VLForConditionalGeneration

    from .qwen2_5_vl.model import Qwen2VLTransformerConfig

    config = Qwen2VLTransformerConfig(
        activation_func=F.silu,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        moe_extended_tp=False,
        perform_initialization=True,
        use_cpu_initialization=True,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        timers=None,
        finalize_model_grads_func=None,
        grad_scale_func=None,
        no_sync_func=None,
        grad_sync_func=None,
        param_sync_func=None,
        deterministic_mode=False,
        enable_autocast=False,
        autocast_dtype=torch.bfloat16,
        num_microbatches_with_partial_activation_checkpoints=None,
        gradient_accumulation_fusion=True,
        pipeline_dtype=torch.bfloat16,
        variable_seq_lengths=False,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
        batch_p2p_sync=True,
        use_ring_exchange_p2p=False,
        deallocate_pipeline_outputs=True,
        defer_embedding_wgrad_compute=False,
        wgrad_deferral_limit=0,
        pipeline_model_parallel_split_rank=None,
        overlap_p2p_comm_warmup_flush=False,
        microbatch_group_size_per_vp_stage=1,
        num_layers=36,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        hidden_size=2048,
        num_attention_heads=16,
        softmax_scale=None,
        num_query_groups=2,
        ffn_hidden_size=11008,
        kv_channels=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        fp32_residual_connection=False,
        apply_residual_connection_post_layernorm=False,
        layernorm_epsilon=1e-06,
        layernorm_zero_centered_gamma=False,
        add_bias_linear=False,
        add_qkv_bias=True,
        gated_linear_unit=True,
        activation_func_fp8_input_store=False,
        num_moe_experts=None,
        rotary_interleaved=False,
        window_size=None,
        normalization="RMSNorm",
        qk_layernorm=False,
        test_mode=False,
        calculate_per_token_loss=False,
        multi_latent_attention=False,
        init_method_std=0.02,
        apply_query_key_layer_scaling=False,
        attention_softmax_in_fp32=False,
        bias_activation_fusion=False,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        memory_efficient_layer_norm=False,
        bias_dropout_fusion=True,
        apply_rope_fusion=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        distribute_saved_activations=False,
        tp_only_amax_red=False,
        cp_comm_type="p2p",
        enable_cuda_graph=False,
        cuda_graph_use_single_mempool=False,
        cuda_graph_retain_backward_graph=False,
        cuda_graph_warmup_steps=2,
        external_cuda_graph=False,
        clone_scatter_output_in_embedding=True,
        disable_parameter_transpose_cache=False,
        config_logger_dir="",
        flash_decode=False,
        inference_rng_tracker=False,
        transformer_impl="transformer_engine",
        rotary_base=1000000,
        rotary_scaling_factor=1.0,
        max_position_embeddings=128000,
        mrope_section=[16, 24, 24],
    )

    return config


def hf_to_mcore_config_llama4(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # Llama4ForConditionalGeneration
    raise NotImplementedError("Llama4ForConditionalGeneration is not supported yet")
