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
from omegaconf import DictConfig

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import RolloutConfig
from verl.workers.megatron_workers import ActorRolloutRefWorker as ARRWorker

# NPU-ADAPTATION: Save the original and dummy copies of `torch.compile`.
from mindspeed.patch_utils import MindSpeedPatchesManager
from mindspeed.core.megatron_basic.requirements_basic import dummy_compile
MindSpeedPatchesManager.patches_info['torch.compile'].remove_patch()
TRUE_COMPILE = torch.compile
DUMMY_COMPILE = dummy_compile
# NPU-ADAPTATION END

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ActorRolloutRefWorker(ARRWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role)

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # NPU-ADAPTATION:Enable TRUE_COMPILEduring rollout graph construction.
        torch.compile = TRUE_COMPILE
        # NPU-ADAPTATION END

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

        if self.config.rollout.name == "vllm":
            from torch.distributed.device_mesh import init_device_mesh

            # NPU-ADAPTATION: Use the adaptation file in this directory.
            from .vllm_rollout_spmd import vLLMRollout
            from .megatron_vllm import MegatronVLLMShardingManager
            # NPU-ADAPTATION END

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
            )
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=rollout_config,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            # NPU-ADAPTATION: Change the input parameter to 'rollout' and replace the 'inference_engine'.
            sharding_manager = MegatronVLLMShardingManager(
                rollout=rollout,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                rollout_config=self.config.rollout,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                bridge=self.bridge,
            )
            # NPU-ADAPTATION END
            log_gpu_memory_usage("After building sharding manager", logger=logger)

            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
            # NPU-ADAPTATION:Enable DUMMY_COMPILE after rollout graph construction.
            torch.compile = DUMMY_COMPILE
            # NPU-ADAPTATION END
        else:
            raise NotImplementedError("Only vllmRollout is supported with Megatron now")
        print(f"rollout and sharding manager init done sharding_manager: {sharding_manager}")
        return rollout, sharding_manager
