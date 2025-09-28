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


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    def __init__(self, config: RewardModelConfig, rm_executor: RewardModelManager = None):
        self.config = config



class RewardManager:
    def __init__(self, config: RewardModelConfig, worker_group: RayWorkerGroup = None):
        self.config = config
        if config.enable and worker_group is not None:
            self.reward_model_manager = RewardModelManager(
                config=config,
                worker_group=worker_group,
            )
            # self.

        assert config.reward_manager == "fapo", "Only fapo reward manager is supported now"

    def compute_rm_score(self, data: DataProto) -> DataProto:
        # if self.config.rollout.free_cache_engine:
        pass

