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
import os

import ray
from hydra import compose, initialize_config_dir
from openai import OpenAI

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager


def test_hybrid_rollout():
    """Test hybrid rollout with context switch."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALL2ALL_BACKEND": "pplx",
            }
        }
    )

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose("ppo_trainer")

    config.trainer.n_gpus_per_node = 4

    # actor config
    model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    # rollout config
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 512
    config.actor_rollout_ref.rollout.response_length = 512
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.7
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False

    # parallelism config
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = int(os.environ["TP"])
    config.actor_rollout_ref.rollout.data_parallel_size = int(os.environ["DP"])
    config.actor_rollout_ref.rollout.expert_parallel_size = int(os.environ["EP"])

    # 1. init hybrid worker: FSDP+rollout
    # - build FSDP model and optimizer
    # - offload FSDP model and optimizer, build rollout
    # - sleep rollout and load FSDP model and optimizer
    agent_loop_manager = init_agent_loop_manager(config)

    # 2. wake up rollout
    # - wake_up weights
    # - load_weights from FSDP
    # - wake_up kv_cache
    agent_loop_manager.wake_up()

    # 3. test async openai call
    server_address = agent_loop_manager.server_addresses[0]
    client = OpenAI(
        api_key="123-abc",
        base_url=f"http://{server_address}/v1",
    )

    smapling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }

    response = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": "What can you do?"}],
        **smapling_params,
    )

    completion = response.choices[0].message.content
    print(f"response: {completion}")
