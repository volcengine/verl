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
import asyncio
import os

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineWorker
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = 8
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen3-VL-2B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl" if get_device_name() == "cuda" else "hccl"

    return config


@pytest.mark.asyncio
async def test_server_adapter(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "VLLM_USE_V1": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
            }
        }
    )

    # 1. create trainer worker group
    model_config: HFModelConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.model)
    checkpoint_engine_config: CheckpointEngineConfig = omega_conf_to_dataclass(
        init_config.actor_rollout_ref.rollout.checkpoint_engine
    )
    trainer_pool = RayResourcePool(process_on_nodes=[4], max_colocate_count=3)
    trainer = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    trainer.reset()

    # 2. create rollout replicas
    rollout_config: RolloutConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.rollout)

    # 2.1 create checkpoint engine worker group
    rollout_pool = RayResourcePool(process_on_nodes=[4], max_colocate_count=3)
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CheckpointEngineWorker),
        model_config=model_config,
        rollout_config=rollout_config,
    )
    rollout = RayWorkerGroup(
        resource_pool=rollout_pool, ray_cls_with_init=ray_cls_with_init, device_name=get_device_name()
    )

    # 2.2 create rollout replicas
    agent_loop_manager = await AgentLoopManager.create(init_config, rollout)

    # 3. create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(
        backend=checkpoint_engine_config.backend, trainer=trainer, replicas=agent_loop_manager.rollout_replicas
    )

    # 4. generate sequences
    raw_prompts = [
        [{"role": "user", "content": "Please write an article about the history of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the history of America, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of America, at least 1000 words."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    n = 4
    batch = batch.repeat(n)
    task = asyncio.create_task(agent_loop_manager.generate_sequences(prompts=batch))

    # 5. update weights to interrupt generate sequences
    for i in range(3):
        await asyncio.sleep(3)
        await checkpoint_manager.update_weights()
        print(f"update weights {i} done")

    # 6. wait for generate sequences task done
    result = await task
    prompts = result.batch["prompts"]
    responses = result.batch["responses"]
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    for i in range(len(result)):
        print("=" * 20)
        print("[PROMPT]", tokenizer.decode(prompts[i], skip_special_tokens=True))
        print("[RESPONSE]", tokenizer.decode(responses[i], skip_special_tokens=True))

    start_model_version = result.non_tensor_batch["start_model_version"]
    finish_model_version = result.non_tensor_batch["finish_model_version"]
    assert np.all(start_model_version == 0), f"start_model_version should be 0, but got {start_model_version}"
    assert np.all(finish_model_version == 3), f"finish_model_version should be 3, but got {finish_model_version}"

    ray.shutdown()
