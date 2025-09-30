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
from contextlib import asynccontextmanager
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
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.utils import get_free_port

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@ray.remote
class SGLangRouter:
    def __init__(self, router_ip: str, router_port: int, worker_urls: list[str], **kwargs):
        from sglang_router.launch_server import RouterArgs, launch_router

        self.router_address = f"{router_ip}:{router_port}"
        router_args = RouterArgs(
            host=router_ip, port=router_port, worker_urls=worker_urls, **kwargs,
        )
        self.router_process = multiprocessing.Process(target=launch_router, args=(router_args,))
        self.router_process.start()

    async def _read_async_response(self, resp: aiohttp.ClientResponse) -> dict[str, Any]:
        if resp.status == 204 or (resp.content_length == 0):
            return {}

        try:
            return await resp.json(content_type=None)
        except Exception:
            try:
                text = await resp.text()
            except Exception:
                return {}
            return {
                "content_type": (resp.headers.get("Content-Type") or ""),
                "text": text,
            }

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        # request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            # "request_id": request_id,
            "image_data": image_data,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            request_url = f"http://{self.router_address}/generate"
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=request_url,
                json=payload,
            ) as response:
                response.raise_for_status()
                return await self._read_async_response(response)
        finally:
            await session.close()
