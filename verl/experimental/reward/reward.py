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
import importlib.util
import multiprocessing
import os
import sys
import queue
import random
import threading
import inspect
from abc import ABC, abstractmethod
from concurrent.futures import Future
from functools import partial
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
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.utils import get_free_port

from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardManager:
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None):
        self.config = config
        self.worker_group = worker_group

    def init_manager(self):
        self._init_reward_model_manager()
        self._init_reward_fn()
        self.reward_model_manager.sleep()

    def _init_reward_model_manager(self):
        if self.config.reward_model.enable:
            self.reward_model_manager = RewardModelManager(
                config=self.config.reward_model,
                worker_group=self.worker_group,
            )
            self.reward_model_manager.sleep()
        else:
            self.reward_model_manager = None

    def _init_reward_fn(self):
        assert self.config.reward_model.reward_manager == "dapo", "Only DAPORewardFunction is supported now."
        from .reward_function import DAPORewardFunction
        self.reward_fn = DAPORewardFunction(self.config, self.reward_model_manager)

    async def compute_score(self, data: DataProto) -> DataProto:
        data_source = data.non_tensor_batch["data_source"].tolist()[0]
        response_ids = data.batch["responses"].tolist()[0]
        ground_truth = data.non_tensor_batch["reward_model"].tolist()[0]["ground_truth"]
        extra_info = data.non_tensor_batch["extra_info"].tolist()[0]
        raw_prompt = data.non_tensor_batch["raw_prompt"].tolist()[0]
        result = await self.reward_fn.run(
            data_source=data_source,
            raw_prompt=raw_prompt,
            response_ids=response_ids,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return result
