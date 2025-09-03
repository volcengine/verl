# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os

os.environ["NCCL_DEBUG"] = "WARN"

from functools import partial

import numpy as np
import pytest
import ray
import torch
from transformers import AutoModelForCausalLM

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import logprobs_from_logits_naive
from verl.workers.config import (
    ActorConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
)
from verl.workers.roles import ActorWorker
from verl.workers.roles.utils.losses import ppo_loss, sft_loss


@pytest.mark.parametrize("strategy", ["megatron", "fsdp", "fsdp2"])
def test_mcore_engine(strategy):
    ray.init()

    path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    model_config = HFModelConfig(path=path)

    if strategy == "megatron":
        engine_config = McoreEngineConfig(
            forward_only=False,
            use_mbridge=False,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
        )
        optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    elif strategy in ["fsdp", "fsdp2"]:
        engine_config = FSDPEngineConfig(
            forward_only=False, fsdp_size=4, strategy=strategy, ulysses_sequence_parallel_size=2
        )
        optimizer_config = FSDPOptimizerConfig()
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    config = ActorConfig(
        model_config=model_config,
        engine=engine_config,
        strategy=strategy,
        ppo_micro_batch_size_per_gpu=256,
        ppo_mini_batch_size=4,
        optim=optimizer_config,
        use_dynamic_bsz=True,
        n=1,
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[8])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # init model
    wg.init_model()

    batch_size = 8
    seqlen = 32

    response_length = seqlen // 2

    torch.manual_seed(1)
    np.random.seed(1)

    input_ids = torch.randint(0, model_config.hf_config.vocab_size, (batch_size, seqlen))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    global_token_num = torch.sum(attention_mask, dim=-1).tolist()

    print(input_ids.float().mean(), attention_mask.float().mean())

    responses = input_ids[:, response_length:]
    response_mask = attention_mask[:, response_length:]

    assert torch.all(response_mask[:, 0] == 1)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num},
    )

    sft_loss_ = partial(sft_loss, config=config)

    # eval
    output = wg.compute_log_prob(data)

    # load hf model and compare results with hf model
    hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    hf_output = hf_model(input_ids, attention_mask=attention_mask)
    hf_logprobs = logprobs_from_logits_naive(
        hf_output.logits[:, -response_length - 1 : -1, :].float(), input_ids[:, -response_length:]
    )
    hf_logprobs_mean = torch.mean(hf_logprobs * response_mask)
    mcore_logprobs_mean = torch.mean(output.batch["old_log_probs"] * response_mask)

    torch.testing.assert_close(hf_logprobs_mean, mcore_logprobs_mean, atol=1e-3, rtol=1e-2)

    data = data.union(output)

    wg.set_loss_fn(sft_loss_)

    # train for one step
    metrics = wg.update_actor(data)
    print(metrics)

    # add ppo data
    data.batch["advantages"] = torch.rand_like(responses, dtype=torch.float32)
    data.batch["ref_log_prob"] = torch.rand_like(responses, dtype=torch.float32)

    # set ppo loss
    ppo_loss_ = partial(ppo_loss, config=config)
    wg.set_loss_fn(ppo_loss_)

    # update again
    ppo_metrics = wg.update_actor(data)
    print(ppo_metrics)

    ray.shutdown()
