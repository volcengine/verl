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

import pytest
import ray
from omegaconf import DictConfig
from openai import AsyncOpenAI

from verl.workers.rollout.replica import get_rollout_replica_class


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 2
    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.load_format = "auto"
    config.actor_rollout_ref.rollout.enforce_eager = True

    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("tp_size", [2, 4])
async def test_standalone_(init_config, tp_size):
    """Test standalone rollout single node and multi nodes."""
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

    init_config.actor_rollout_ref.rollout.skip_tokenizer_init = False
    init_config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
    num_replicas = (init_config.trainer.n_gpus_per_node * init_config.trainer.nnodes) // tp_size

    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(init_config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(replica_rank=replica_rank, config=init_config, gpus_per_node=2)
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("NO_PROXY", None)

    client = AsyncOpenAI(
        api_key="123-abc",
        base_url=f"http://{server_addresses[0]}/v1",
    )

    completion = await client.chat.completions.create(
        model=init_config.actor_rollout_ref.model.path,
        messages=[{"role": "user", "content": "What can you do?"}],
    )
    print(completion.choices[0].message.content)

    ray.shutdown()
