# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import os

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
)
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader
from verl.workers.fsdp_workers import ActorRolloutRefWorker as Workder
from verl.workers.fsdp_workers import CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["ActorRolloutRefWorker", "AsyncActorRolloutRefWorker", "CriticWorker"]


class ActorRolloutRefWorker(Workder):
    def _build_rollout(self, trust_remote_code=False):
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm"

        from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        assert vllm_mode == "spmd"

        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}

        from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

        vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
        rollout = vllm_rollout_cls(model_path=local_path, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, device_mesh=rollout_device_mesh, trust_remote_code=trust_remote_code, **lora_kwargs)
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        if self.config.hybrid_engine:
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
        else:
            from .vllm_sharding_manager import VLLMShardingManager

            rollout.inference_engine.wake_up()
            rollout_sharding_manager = VLLMShardingManager(inference_engine=rollout.inference_engine, device_mesh=rollout_device_mesh)

            log_gpu_memory_usage("After building sharding manager", logger=logger)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        # fsdp module and checkpoint manager are unnecessary when hybrid_engine is disabled
        if self._is_rollout and not self.config.hybrid_engine:
            del self.actor_module_fsdp
            del self.actor_optimizer
            del self.actor_lr_scheduler
            del self.checkpoint_manager
            log_gpu_memory_usage("After delete actor model during init", logger=logger)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def generate_sequences(self, *args, **kwargs):
        return super().generate_sequences(*args, **kwargs)

    def _get_actor_params(self):
        assert self._is_actor
        params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp))
        return params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        params = self._get_actor_params() if self._is_actor else None
        if self._is_rollout:
            inference_model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(inference_model)
        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)
            from ray.util.collective import collective

            collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
            if self._is_rollout:
                inference_model.load_weights([(key, tensor)])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        self._weights_info = ret
        return ret


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
