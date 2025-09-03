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
"""
The main entry point to run the PPO algorithm
"""

import logging
import os

import torch
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.ppo import core_algos
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_nccl_backend,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.engine import EngineRegistry
from verl.workers.reward_model.sglang_reward import SGLangReward
from verl.workers.config import HFModelConfig


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config, model_config: HFModelConfig) -> None:
        self.config = config
        self.model_config = model_config
        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.reward_model = SGLangReward(
            actor_module=self.model_config.local_path,
            config=self.config,
            processing_class=self.model_config.get_processor(),
            model_hf_config=self.model_config.hf_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def _switch_chat_template(self, data: DataProto):
        pass

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        self.reward_model.compute_reward(data)
