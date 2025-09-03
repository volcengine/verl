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
    get_device_name,
    get_torch_device,
    get_nccl_backend,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.engine import EngineRegistry
from verl.workers.reward_model.sglang_reward import SGLangRewardModel
from verl.workers.config import RewardModelConfig, HFModelConfig


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class RewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config: RewardModelConfig) -> None:
        self.config = config
        self.model_config = config.model_config
        self.actor_model_config = config.actor_model_config
        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

    def _build_reward_model(self):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse reward model and huggingface model config
        reward_model_config: RewardModelConfig = self.config
        model_config: HFModelConfig = self.config.model_config

        # 2. build reward model device mesh
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"reward model world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        reward_model_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        is_collect = reward_model_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "reward_model", dp_rank=reward_model_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and reward model random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = reward_model_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build reward model
        log_gpu_memory_usage(f"Before building sglang reward model", logger=logger)
        self.reward_model = SGLangRewardModel(
            config=reward_model_config, model_config=model_config, device_mesh=reward_model_device_mesh
        )
        log_gpu_memory_usage(f"After building sglang reward model", logger=logger)        


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_reward_model()

    def _switch_chat_template(self, data: DataProto):
        pass

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        breakpoint()
        self.reward_model.compute_reward(data)
