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
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayResourcePool, ResourcePoolManager
from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig

logger = logging.getLogger(__file__)


class TokenOutput(BaseModel):
    token_ids: list[int]
    """response token ids"""
    log_probs: Optional[list[float]] = None
    """logprobs of response token ids"""


# Worker class mapping for different rollout mode:
# +-------------------------+---------------------------+-----------------------------+
# |                         |         vLLM              |          SGLang             |
# +-------------------------+---------------------------+-----------------------------+
# | batch mode, hybrid      | ActorRolloutRefWorker     | ActorRolloutRefWorker       |
# +-------------------------+---------------------------+-----------------------------+
# | server mode, hybrid     | AsyncActorRolloutRefWorker| AsyncActorRolloutRefWorker  |
# +-------------------------+---------------------------+-----------------------------+
# | server mode,            | vLLMAsyncRollout          | SGLangRollout               |
# | colocated/standalone    |                           |                             |
# +-------------------------+---------------------------+-----------------------------+
#
# Note: the batch mode is going to be deprecated.


class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"


class RolloutServer(ABC):
    """Rollout server is an abstraction of individual server instance, responsible for managing
    the lifecycle of server, including initialization, context switch, weight sync, etc.

    Args:
        dp_rank: int, data parallel rank of this rollout server.
        config: RolloutConfig, rollout server config.
        model_config: HFModelConfig, model config.
    """

    def __init__(self, dp_rank: int, config: RolloutConfig, model_config: HFModelConfig) -> None:
        self.dp_rank = dp_rank
        self.config: RolloutConfig = config
        self.model_config: HFModelConfig = model_config

        self.mode: RolloutMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: RayResourcePool = None

        self._server_address: str = None
        self._server_handle: ActorHandle = None

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.mode = RolloutMode.HYBRID
        tp_size = self.config.tensor_model_parallel_size
        self.workers = worker_group.workers[tp_size * self.dp_rank : tp_size * (self.dp_rank + 1)]
        await self.init_server()

    async def init_colocated(self, resource_pool: RayResourcePool):
        """Init colocated rollout server, rollout engine and hybrid engine colocated in same ray placement group
        but in separate processes.

        Args:
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
        """
        self.mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool

        # FIXME(@wuxibin): create worker group for this rollout
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_colocated_{self.dp_rank}",
        )
        self.workers = worker_group.workers
        await self.init_server()

    async def init_standalone(self):
        """Init standalone rollout server, create new resource pool for this rollout."""
        # create resource pool for this rollout
        assert self.config.load_format == "auto", "standalone rollout load_format should be auto."

        self.mode = RolloutMode.STANDALONE
        resource_pool_spec = {
            f"rollout_pool_{self.dp_rank}": [self.config.tensor_model_parallel_size],
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[f"rollout_pool_{self.dp_rank}"]

        # create worker group for this rollout
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_standalone_{self.dp_rank}",
        )
        self.workers = worker_group.workers
        await self.init_server()

    @abstractmethod
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class."""
        raise NotImplementedError

    @abstractmethod
    async def init_server(self):
        """Init rollout server."""
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        """Get rollout server address for OpenAI chat completion."""
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        """Get rollout server handle for Token-in-token-out generation."""
        return self._server_handle

    @abstractmethod
    async def wake_up(self):
        """Wake up rollout server."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep rollout server."""
        raise NotImplementedError

    @abstractmethod
    async def update_weights(self):
        """Update weights of standalone rollout server."""
        raise NotImplementedError


def get_rollout_server_class(rollout: str):
    if rollout == "vllm":
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMRolloutServer

        return vLLMRolloutServer
    elif rollout == "sglang":
        from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangRolloutServer

        return SGLangRolloutServer
    else:
        raise ValueError(f"Unknown rollout mode: {rollout}")
