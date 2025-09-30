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
import aiohttp
import heapq
import logging
import multiprocessing
import os
import queue
import random
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayResourcePool
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.utils import get_free_port

from .sglang_router import SGLangRouter

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    def __init__(self, config: RewardModelConfig, worker_group: RayWorkerGroup = None):
        self.config = config
        self.worker_group = worker_group
        self._initialize_llm_servers()
        self._initialize_router()

    def _initialize_llm_servers(self):
        assert self.config.rollout.name == "sglang", "Only sglang is supported now"
        rollout_world_size = self.config.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(self.config.rollout.name)
        rollout_config = self.config.rollout
        model_config = HFModelConfig(
            path=self.config.model.path,
            external_lib=self.config.model.external_lib,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_colocated(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_router(self):
        router_ip = ray.util.get_node_ip_address()
        router_port, _ = get_free_port(router_ip)

        # current implementation only support sglang
        assert self.config.rollout.name == "sglang", "Only sglang is supported now"
        router = SGLangRouter(router_ip, router_port, self.server_addresses, balance_abs_threshold=4)
        self.router = router

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        return asyncio.create_task(run_all())

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        return await self.router.generate(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
        )

    def generate_sequences(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        pass