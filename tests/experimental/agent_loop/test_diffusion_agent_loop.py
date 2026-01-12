# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.diffusion_agent_loop import DiffusionAgentLoopManager
from verl.protocol import DataProto


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    model_path = os.path.expanduser("~/models/Qwen/Qwen-Image")
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm-omni"
    config.actor_rollout_ref.rollout.n = 4

    return config


def test_single_turn(init_config):
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

    prompt = (
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," '
        'with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful '
        'Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". '
        "Ultra HD, 4K, cinematic composition"
    )

    agent_loop_manager = DiffusionAgentLoopManager(init_config)

    raw_prompts = [[{"role": "user", "content": prompt}]]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["diffusion_single_turn_agent"] * len(raw_prompts)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    if init_config.actor_rollout_ref.rollout.calculate_log_probs:
        assert result.batch["rollout_log_probs"].size(1) == result.batch["responses"].size(1)

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()
