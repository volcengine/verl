# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import json
import logging
import os

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer

from recipe.swe_agent.eval_on_vefaas_test import load_instance

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/trainer/config")
    print(config_dir)
    with initialize_config_dir(config_dir=config_dir):
        config = compose(config_name="ppo_trainer")
    model_path = "/mnt/hdfs/wuxibin_wl/model/Qwen3-Coder-30B-A3B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "qwen3_xml"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 16384
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.6
    config.actor_rollout_ref.rollout.agent.num_workers = 1

    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_react_agent(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    agent_loop_config = [
        {
            "_target_": "recipe.swe_agent.swe_agent_loop.SWEAgentLoop",
            "name": "react_agent",
        },
    ]
    agent_loop_config_path = "/tmp/agent_loop_config.json"
    with open(agent_loop_config_path, "w") as f:
        json.dump(agent_loop_config, f)

    n = 1
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
    init_config.actor_rollout_ref.rollout.agent.agent_loop_config_path = agent_loop_config_path
    agent_loop_manager = init_agent_loop_manager(init_config)

    # =========================== 2. Generate sequences  ===========================
    instance_id = "django__django-15368"
    subnet = "verified"
    metadata, image = load_instance(instance_id, subnet)

    samples = [
        {
            "raw_prompt": [
                {
                    "role": "user",
                    "content": "",
                },
            ],
            "agent_name": "react_agent",
            "extra_info": {
                "tools_kwargs": {
                    "dataset_id": "verified", 
                    "instance_id": instance_id, 
                    "metadata": metadata, 
                    "image": image,
                }
            },
        },
    ]

    batch = DataProto(
        non_tensor_batch={
            "data_source": np.array(["swe_agent" for _ in range(len(samples))], dtype=object),
            "raw_prompt": np.array([np.array(sample["raw_prompt"]) for sample in samples], dtype=object),
            "agent_name": np.array([sample["agent_name"] for sample in samples], dtype=object),
            "tools_kwargs": np.array([sample["extra_info"]["tools_kwargs"] for sample in samples], dtype=object),
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(samples) * n
    assert "rm_scores" in result.batch

    # Check response_mask
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"
    response_length = response_mask.size(1)

    for i in range(len(responses)):
        # response with tool response
        valid_tokens = responses[i][attention_mask[i][-response_length:].bool()]
        response_with_obs = tokenizer.decode(valid_tokens)

        print("=========================")
        print(response_with_obs)

    print("Test passed!")
    ray.shutdown()
