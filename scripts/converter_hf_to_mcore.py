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

from typing import List, Tuple, Dict
import re
import os
import torch
import argparse
import warnings
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file
from torch.distributed._tensor import Shard, Placement
from verl.utils.megatron_utils import get_model, convert_config
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core import parallel_state as mpu
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import StrictHandling


def _init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_path', type=str, required=True, help="The path for the huggingface model")
    parser.add_argument('--output_path', type=str, required=True, help="The path for the output mcore model")
    parser.add_argument('--test', action='store_true', help="Whether to test the conversion")
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


def convert_hf_to_mcore(hf_model_path, output_path, test=False):
    os.makedirs(output_path, exist_ok=True)
    if len(os.listdir(output_path)) > 0 and not test:
        print(f"Output path {output_path} is not empty, skipping conversion")
        return

    # init torch distributed and mpu
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group('nccl')
    mpu.initialize_model_parallel(tensor_model_parallel_size=1,
                                  virtual_pipeline_model_parallel_size=None,
                                  context_parallel_size=1,
                                  expert_model_parallel_size=1)

    # init hf config
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    print(hf_config)
    megatron_config = MegatronConfig()
    cfg = Config()
    cfg.model.path = hf_model_path
    tfconfig = convert_config(hf_config, megatron_config)
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # init megatron model
    def megatron_model_provider(pre_process, post_process):
        from verl.utils.model import get_parallel_gptmodel_from_config
        parallel_model = get_parallel_gptmodel_from_config(tfconfig,
                                                           hf_config,
                                                           pre_process,
                                                           post_process,
                                                           share_embeddings_and_output_weights=tie_word_embeddings,
                                                           value=False)
        return parallel_model

    model = get_model(model_provider_func=megatron_model_provider,
                      model_type=ModelType.encoder_or_decoder,
                      wrap_with_ddp=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # init hf model
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    ref_state_dict = hf_model.state_dict()

    # load hf state dict to megatron model
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel
    load_state_dict_to_megatron_gptmodel(state_dict=ref_state_dict,
                                         wrapped_models=model,
                                         config=hf_config,
                                         params_dtype=torch.bfloat16,
                                         is_value_model=False)
    ssd = model[0].module.module.sharded_state_dict()
    del ref_state_dict, hf_model

    # save megatron model
    if len(os.listdir(output_path)) == 0:
        dist_checkpointing.save(ssd, output_path, sharded_strategy=None, async_sharded_save=False)
    if test:
        ########### test ###########
        # load model
        model_test = get_model(model_provider_func=megatron_model_provider,
                               model_type=ModelType.encoder_or_decoder,
                               wrap_with_ddp=True)
        ssd2 = model_test[0].module.module.sharded_state_dict()
        dist_checkpointing.load(ssd2, output_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

        sd = model[0].module.module.state_dict()
        sd2 = model_test[0].module.module.state_dict()
        for k in sd.keys():
            if sd[k] is None:
                continue
            d1 = sd[k].data
            if k in sd2:
                d2 = sd2[k].data
                assert d1.shape == d2.shape, f'{k=} {d1.shape=} {d2.shape=}'
                assert (d1 == d2).all(), f"{k} is not equal"
        for k in sd2.keys():
            if sd2[k] is None:
                continue
            d1 = sd2[k].data
            if k in sd:
                d2 = sd[k].data
                assert d1.shape == d2.shape, f'{k=} {d1.shape=} {d2.shape=}'
                assert (d1 == d2).all(), f"{k} is not equal"

        # load value model
        def megatron_value_model_provider(pre_process, post_process):
            from verl.utils.model import get_parallel_gptmodel_from_config
            parallel_model = get_parallel_gptmodel_from_config(tfconfig,
                                                               hf_config,
                                                               pre_process,
                                                               post_process,
                                                               share_embeddings_and_output_weights=False,
                                                               value=True)
            parallel_model.cuda()
            return parallel_model

        model_value = get_model(model_provider_func=megatron_value_model_provider,
                                model_type=ModelType.encoder_or_decoder,
                                wrap_with_ddp=True)
        ssd2 = model_value[0].module.module.sharded_state_dict()
        dist_checkpointing.load(ssd2, output_path, strict=StrictHandling.IGNORE_ALL)

        sd = model[0].module.module.state_dict()
        sd2 = model_value[0].module.module.state_dict()
        for k in sd.keys():
            if sd[k] is None:
                continue
            d1 = sd[k].data
            if k in sd2:
                d2 = sd2[k].data
                assert d1.shape == d2.shape, f'{k=} {d1.shape=} {d2.shape=}'
                assert (d1 == d2).all(), f"{k} is not equal"
        for k in sd2.keys():
            if sd2[k] is None:
                continue
            d1 = sd2[k].data
            if k in sd:
                d2 = sd[k].data
                assert d1.shape == d2.shape, f'{k=} {d1.shape=} {d2.shape=}'
                assert (d1 == d2).all(), f"{k} is not equal"


if __name__ == "__main__":
    args = _init_args()
    convert_hf_to_mcore(args.hf_model_path, args.output_path, args.test)
