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

import argparse
import os
import warnings

import torch
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.serialization import StrictHandling
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig, AutoModelForCausalLM

from verl.models.mcore import hf_to_mcore_config
from verl.utils.megatron_utils import get_model


def _init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
    parser.add_argument("--output_path", type=str, required=True, help="The path for the output mcore model")
    parser.add_argument("--test", action="store_true", help="Whether to test the conversion")
    args = parser.parse_args()
    return args


class MegatronConfig:
    def __init__(self):
        self.params_dtype = torch.bfloat16


class ModelConfig:
    def __init__(self):
        self.path = None


class Config:
    def __init__(self):
        self.model = ModelConfig()


def convert_checkpoint_from_transformers_to_megatron(hf_model, model, hf_config):
    num_attention_heads = hf_config.num_attention_heads
    hidden_dim = hf_config.hidden_size
    head_dim = hidden_dim // num_attention_heads
    with torch.no_grad():
        model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)
        for layer, hf_layer in zip(model.decoder.layers, hf_model.model.layers):
            layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.input_layernorm.weight)

            q = hf_layer.self_attn.q_proj.weight.view([num_attention_heads, -1, head_dim, hidden_dim])
            k = hf_layer.self_attn.k_proj.weight.view([num_attention_heads, -1, head_dim, hidden_dim])
            v = hf_layer.self_attn.v_proj.weight.view([num_attention_heads, -1, head_dim, hidden_dim])
            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()

            q_bias = hf_layer.self_attn.q_proj.bias.view([num_attention_heads, -1])
            k_bias = hf_layer.self_attn.k_proj.bias.view([num_attention_heads, -1])
            v_bias = hf_layer.self_attn.v_proj.bias.view([num_attention_heads, -1])
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()

            layer.self_attention.linear_qkv.weight.copy_(qkv)
            layer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            layer.self_attention.linear_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)

            layer.mlp.router.weight.copy_(hf_layer.mlp.gate.weight)

            for idx, hf_expert in enumerate(hf_layer.mlp.experts):
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                layer.mlp.experts.linear_fc1._parameters[f"weight{idx}"].copy_(fc1_weight)
                layer.mlp.experts.linear_fc2._parameters[f"weight{idx}"].copy_(hf_expert.down_proj.weight)

            layer.mlp.shared_experts.gate_weight.copy_(hf_layer.mlp.shared_expert_gate.weight)
            shared_fc1_weight = torch.cat([hf_layer.mlp.shared_expert.gate_proj.weight, hf_layer.mlp.shared_expert.up_proj.weight])
            layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
            layer.mlp.shared_experts.linear_fc2.weight.copy_(hf_layer.mlp.shared_expert.down_proj.weight)

        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        model.output_layer.weight.copy_(hf_model.lm_head.weight)


def convert_hf_to_mcore(hf_model_path, output_path, test=False):
    os.makedirs(output_path, exist_ok=True)
    if len(os.listdir(output_path)) > 0 and not test:
        print(f"Output path {output_path} is not empty, skipping conversion")
        return

    # init torch distributed and mpu
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

    # init hf config
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    print(hf_config)

    cfg = Config()
    cfg.model.path = hf_model_path
    tfconfig = hf_to_mcore_config(hf_config, torch.bfloat16)
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # init megatron model
    def megatron_model_provider(pre_process, post_process):
        from verl.models.mcore import init_mcore_model

        parallel_model = init_mcore_model(
            tfconfig,
            hf_config,
            pre_process,
            post_process,
            share_embeddings_and_output_weights=tie_word_embeddings,
            value=False,
        )
        return parallel_model

    model = get_model(model_provider_func=megatron_model_provider, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # init hf model
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16)
    ref_state_dict = hf_model.state_dict()

    # load hf state dict to megatron model
    if "Qwen2MoeForCausalLM" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    else:
        from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

        load_state_dict_to_megatron_gptmodel(
            state_dict=ref_state_dict,
            wrapped_models=model,
            config=hf_config,
            params_dtype=torch.bfloat16,
            is_value_model=False,
        )

    ssd = model[0].module.sharded_state_dict()
    del ref_state_dict, hf_model

    # save megatron model
    if len(os.listdir(output_path)) == 0:
        dist_checkpointing.save(ssd, output_path, sharded_strategy=None, async_sharded_save=False)
    if test:
        ########### test ###########
        # load model
        model_test = get_model(model_provider_func=megatron_model_provider, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True)
        ssd2 = model_test[0].module.sharded_state_dict()
        dist_checkpointing.load(ssd2, output_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

        sd = model[0].module.state_dict()
        sd2 = model_test[0].module.state_dict()
        for k in sd.keys():
            if sd[k] is None:
                continue
            d1 = sd[k].data
            if k in sd2:
                d2 = sd2[k].data
                assert d1.shape == d2.shape, f"{k=} {d1.shape=} {d2.shape=}"
                assert (d1 == d2).all(), f"{k} is not equal"
        for k in sd2.keys():
            if sd2[k] is None:
                continue
            d1 = sd2[k].data
            if k in sd:
                d2 = sd[k].data
                assert d1.shape == d2.shape, f"{k=} {d1.shape=} {d2.shape=}"
                assert (d1 == d2).all(), f"{k} is not equal"

        # load value model
        def megatron_value_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_gptmodel_from_config

            parallel_model = get_parallel_gptmodel_from_config(tfconfig, hf_config, pre_process, post_process, share_embeddings_and_output_weights=False, value=True)
            parallel_model.cuda()
            return parallel_model

        model_value = get_model(
            model_provider_func=megatron_value_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
        )
        ssd2 = model_value[0].module.sharded_state_dict()
        dist_checkpointing.load(ssd2, output_path, strict=StrictHandling.IGNORE_ALL)

        sd = model[0].module.state_dict()
        sd2 = model_value[0].module.state_dict()
        for k in sd.keys():
            if sd[k] is None:
                continue
            d1 = sd[k].data
            if k in sd2:
                d2 = sd2[k].data
                assert d1.shape == d2.shape, f"{k=} {d1.shape=} {d2.shape=}"
                assert (d1 == d2).all(), f"{k} is not equal"
        for k in sd2.keys():
            if sd2[k] is None:
                continue
            d1 = sd2[k].data
            if k in sd:
                d2 = sd[k].data
                assert d1.shape == d2.shape, f"{k=} {d1.shape=} {d2.shape=}"
                assert (d1 == d2).all(), f"{k} is not equal"


if __name__ == "__main__":
    args = _init_args()
    convert_hf_to_mcore(args.hf_model_path, args.output_path, args.test)
