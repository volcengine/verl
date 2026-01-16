# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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
#
# Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/engine/core.py

import logging
import os
import time

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _initialize_kv_caches(self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
    start = time.time()

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # Profiles the peak memory usage of the model to determine how much
    # memory can be allocated for kv cache.
    available_gpu_memory = self.model_executor.determine_available_memory()

    assert len(kv_cache_specs) == len(available_gpu_memory)
    # Get the kv cache tensor size
    self.kv_cache_configs = [
        get_kv_cache_config(vllm_config, kv_cache_spec_one_worker, available_gpu_memory_one_worker)
        for kv_cache_spec_one_worker, available_gpu_memory_one_worker in zip(
            kv_cache_specs, available_gpu_memory, strict=False
        )
    ]

    # Since we use a shared centralized controller, we need the
    # `kv_cache_config` to be consistent across all workers to make sure
    # all the memory operators can be applied to all workers.
    unify_kv_cache_configs(self.kv_cache_configs)

    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    assert all([cfg.num_blocks == self.kv_cache_configs[0].num_blocks for cfg in self.kv_cache_configs])
    num_gpu_blocks = self.kv_cache_configs[0].num_blocks
    num_cpu_blocks = 0
    scheduler_kv_cache_config = self.kv_cache_configs[0]

    # Initialize kv cache and warmup the execution
    self.model_executor.initialize_from_config(self.kv_cache_configs)

    elapsed = time.time() - start
    logger.info(("init engine (profile, create kv cache, warmup model) took %.2f seconds"), elapsed)
    return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config


EngineCore._initialize_kv_caches = _initialize_kv_caches
