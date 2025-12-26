# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

import multiprocessing
import os

import psutil
import pytest
import ray
import uvicorn
from omegaconf import DictConfig

from recipe.agent_lightning_like.example.agent_server import HEADER_TRACE_ID, _get_free_port, _get_host_ip
from recipe.agent_lightning_like.llm_router import LLMRouter
from recipe.agent_lightning_like.notify import wait_for_server
from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.utils import hf_tokenizer


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("recipe/agent_lightning_like/config"), version_base=None):
        config = compose(
            config_name="lightning_ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )
    config.trainer.n_gpus_per_node = 2
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-3B-Instruct"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "sglang")
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.6
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 2048

    config.data.max_prompt_length = 1024
    config.data.return_raw_chat = True

    config.actor_rollout_ref.rollout.agent.num_workers = 2
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = (
        "recipe/agent_lightning_like/example/agent_loop.yaml"
    )

    config.lightning_trainer.agent_server_addr = None  # to be filled after the agent server started
    config.lightning_trainer.agent_client_config_path = "recipe/agent_lightning_like/example/agent_client.yaml"
    config.lightning_trainer.request_header_trace_id = HEADER_TRACE_ID

    return config


def run_agent_server(port: int, num_workers=16) -> None:
    import sys

    sys.stdin = None
    uvicorn.run(
        "recipe.agent_lightning_like.example.agent_server:app",
        host="0.0.0.0",
        port=port,
        workers=num_workers,
        log_level="warning",
    )


def shutdown_agent_server(pid: int):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


@pytest.fixture
def agent_server_addr():
    if os.environ.get("LLM_SERVER_NOTIFY_FILE") is None:
        ## test only. in production, use a shared file system,
        ## or a more robust way to notify the llm server address.
        os.environ["LLM_SERVER_NOTIFY_FILE"] = "/tmp/llm_server_notify"

    localhost = _get_host_ip()
    agent_server_port = _get_free_port()
    agent_server_addr = f"{localhost}:{agent_server_port}"

    p_agent_server = multiprocessing.Process(target=run_agent_server, args=(agent_server_port,))
    p_agent_server.start()
    wait_for_server(agent_server_addr, "/health")

    yield agent_server_addr

    shutdown_agent_server(p_agent_server.pid)
    try:
        os.remove(os.environ["LLM_SERVER_NOTIFY_FILE"])
    except Exception:
        pass


@pytest.fixture
def agent_loop_mgr(init_config, agent_server_addr):
    env_vars = {
        "NCCL_DEBUG": "WARN",
    }
    ray.init(runtime_env={"env_vars": env_vars})

    assert agent_server_addr
    config = init_config
    config.lightning_trainer.agent_server_addr = agent_server_addr
    manager = init_agent_loop_manager(config)

    yield manager

    ray.shutdown()


@pytest.fixture
def tokenizer(init_config):
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    return tokenizer


@pytest.fixture
def llm_router_and_addr(init_config, agent_loop_mgr, tokenizer) -> tuple[LLMRouter, str]:
    llm_router = LLMRouter.options(
        name="LLMRouter",  # name required for ray.get_actor later
    ).remote(
        config=init_config,
        tokenizer=tokenizer,
        server_handles=agent_loop_mgr.server_handles,
    )
    address = ray.get(llm_router.get_server_address.remote())
    wait_for_server(address, "/v1/models")

    return llm_router, address
