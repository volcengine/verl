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

# use mcore transformer config to initialize the model


def init_mcore_model_dense(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    # for LlamaForCausalLM, Qwen2ForCausalLM
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    use_te = True
    assert tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
    transformer_layer_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
    rope_scaling_args = {}
    if hf_config.rope_scaling is not None:
        assert hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
        rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling["factor"]
    model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rotary_base=hf_config.rope_theta,
        **rope_scaling_args,
    )
    if post_process and value:
        from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

        model.output_layer = LinearForLastLayer(input_size=tfconfig.hidden_size, output_size=1, config=tfconfig)
    return model


def init_mcore_model_qwen2_moe(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    return init_mcore_model_dense(
        tfconfig, hf_config, pre_process, post_process, share_embeddings_and_output_weights, value
    )


def init_mcore_model_llama4(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    return init_mcore_model_dense(
        tfconfig, hf_config, pre_process, post_process, share_embeddings_and_output_weights, value
    )


def init_mcore_model_dpskv3(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    return init_mcore_model_dense(
        tfconfig, hf_config, pre_process, post_process, share_embeddings_and_output_weights, value
    )


def init_mcore_model_qwen2_5_vl(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    # Qwen2_5_VLForConditionalGeneration
    raise NotImplementedError("VLM is not supported yet")
