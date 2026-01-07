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

import ray
import torch
from transformers import AutoModelForCausalLM

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.fs import copy_to_local
from verl.workers.config import FSDPEngineConfig, HFModelConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig


class TrainingWorkerTest(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, checkpoint_backend: str, checkpoint_kwargs: dict) -> None:
        copy_to_local(config.model_config.path)
        super().__init__(config)
        if torch.distributed.get_rank() == 0 and checkpoint_backend == "nccl":
            checkpoint_kwargs["is_master"] = True
        self.checkpoint_engine = CheckpointEngineRegistry.new(checkpoint_backend, **checkpoint_kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


class RolloutWorkerTest:
    def __init__(
        self,
        model_path,
        checkpoint_backend: str,
        checkpoint_kwargs: dict,
        device: str = "cuda",
        check_allclose: bool = True,
    ) -> None:
        self.checkpoint_engine = CheckpointEngineRegistry.new(checkpoint_backend, **checkpoint_kwargs)
        local_path = copy_to_local(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16)
        self.model.to(device)
        self.check_allclose = check_allclose
        self.received_weights: dict[str, torch.Tensor] = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        async for name, weight in self.checkpoint_engine.receive_weights():
            weight = weight.clone()
            if self.check_allclose:
                self.received_weights[name] = weight.clone().to(torch.bfloat16)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_weights(self):
        if not self.check_allclose:
            return
        for name, weight in self.model.state_dict().items():
            assert name in self.received_weights, f"weight {name} not received"
            assert torch.allclose(weight, self.received_weights[name]), f"weight {name} not equal"
        self.received_weights.clear()


def create_trainer_worker_group(
    model_path: str, resource_pool: RayResourcePool, checkpoint_backend: str, checkpoint_kwargs: dict
) -> RayWorkerGroup:
    local_path = copy_to_local(model_path)
    model_config = HFModelConfig(path=local_path, use_remove_padding=True)
    engine_config = FSDPEngineConfig(forward_only=True, fsdp_size=resource_pool.world_size, strategy="fsdp")

    trainer_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
    )
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(TrainingWorkerTest),
        config=trainer_config,
        checkpoint_backend=checkpoint_backend,
        checkpoint_kwargs=checkpoint_kwargs,
    )
    ray_cls_with_init.update_options(
        {
            "runtime_env": {
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        }
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    return wg


def create_rollout_worker_group(
    model_path: str, resource_pool: RayResourcePool, checkpoint_backend: str, checkpoint_kwargs: dict, **kwargs
) -> RayWorkerGroup:
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(RolloutWorkerTest),
        model_path=model_path,
        checkpoint_backend=checkpoint_backend,
        checkpoint_kwargs=checkpoint_kwargs,
        **kwargs,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    return wg
