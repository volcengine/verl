# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import (
    log_gpu_memory_usage,
)
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader
from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.workers.megatron_workers import CriticWorker, RewardModelWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["DetachActorWorker", "DetachRolloutWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


class DetachNcclSync(ActorRolloutRefWorker):
    def _get_actor_params_generator(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        params_generator = self._get_actor_params_generator() if self._is_actor else None
        if self._is_rollout:
            inference_model = (
                self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            )
            patch_vllm_moe_model_weight_loader(inference_model)
        for key, shape, dtype in self._weights_info:
            if self._is_actor:
                weight_key, weight = next(params_generator)
                assert key == weight_key
                assert shape == weight.size()
                assert dtype == weight.dtype

            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor and torch.distributed.get_rank() == 0:
                tensor.copy_(weight)
            from ray.util.collective import collective

            collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")
            if self._is_rollout:
                inference_model.load_weights([(key, tensor)])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info

        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        return ret


class DetachActorWorker(DetachNcclSync):
    def _get_actor_params_generator(self):
        assert self._is_actor
        from verl.models.mcore import get_mcore_weight_converter
        from verl.utils.megatron_utils import per_tensor_generator

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        generator = per_tensor_generator(
            self.actor.actor_module,
            self.actor_model_config,
            weight_converter,
            self.tf_config,
            layer_name_mapping,
        )
        return generator


class DetachRolloutWorker(DetachNcclSync):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = {}
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        from verl.utils.model import get_generation_config

        self._init_hf_config_and_tf_config(
            self.config.model.path,
            self.config.model.path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            trust_remote_code,
        )
        self.generation_config = get_generation_config(self.local_path)

        from torch.distributed.device_mesh import init_device_mesh

        assert self.config.rollout.name == "vllm"

        from verl.workers.rollout.vllm_rollout import vLLMRollout

        # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
        # we will reorganize their weight format when resharding from actor to rollout.

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
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.hf_config,
            device_mesh=rollout_device_mesh,
            trust_remote_code=trust_remote_code,
        )
        log_gpu_memory_usage("After building vllm rollout", logger=logger)

        from sharding_manager import DetachShardingManager
        rollout_sharding_manager = DetachShardingManager(
            inference_engine=rollout.inference_engine, device_mesh=rollout_device_mesh
        )

        log_gpu_memory_usage("After building sharding manager", logger=logger)

        self.rollout = rollout
        self.sharding_manager = rollout_sharding_manager
        self.rollout.sharding_manager = rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def async_generate_sequences(self, *args, **kwargs):
        return super().generate_sequences(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info


class DetachAsyncRolloutWorker(AsyncActorRolloutRefWorker, DetachRolloutWorker):
    def __init__(self, config: DictConfig, role: str):
        print(DetachAsyncRolloutWorker.__mro__)
        DetachRolloutWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        DetachRolloutWorker.init_model(self)