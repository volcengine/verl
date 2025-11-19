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

import asyncio
import logging
import os
import threading
import time

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    fsdp_version,
)
from verl.utils.ray_utils import get_event_loop
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker"]


def get_inference_model(rollout):
    """
    get models according to different types of inference_engine
    Args:
        rollout: rollout object
    Returns:
        model: model object
    """
    inference_engine = rollout.inference_engine
    if hasattr(inference_engine, "llm_engine"):
        inference_model = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    elif hasattr(inference_engine, "worker"):
        inference_model = inference_engine.worker.model_runner.model
    else:
        raise AttributeError(
            f"Unsupported inference_engine type: {type(inference_engine)}. "
            f"Expected LLM (with llm_engine attribute) or WorkerWrapperBase (with worker attribute)."
        )
    return inference_model


class DetachNcclSync(AsyncActorRolloutRefWorker):
    def _get_actor_params(self):
        pass

    def _run_async_safely(self, coro):
        """Run async coroutine safely, handling cases where event loop may already be running.
        
        Uses a new event loop in a separate thread when the current thread already has a running loop.
        """
        try:
            # Check if there's a running event loop in the current thread
            loop = asyncio.get_running_loop()
            # Event loop is already running, use a new loop in a separate thread
            tmp_event_loop = asyncio.new_event_loop()
            
            def run_loop():
                asyncio.set_event_loop(tmp_event_loop)
                tmp_event_loop.run_forever()
            
            thread = threading.Thread(
                target=run_loop,
                name="async_weight_sync",
                daemon=True,
            )

            def run_coroutine(coroutine):
                if not thread.is_alive():
                    thread.start()
                    # Give the thread a moment to start the loop
                    time.sleep(0.01)
                future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
                return future.result()

            async def stop_loop():
                tmp_event_loop.stop()

            try:
                return run_coroutine(coro)
            finally:
                if thread.is_alive():
                    asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
                    thread.join(timeout=1.0)
        except RuntimeError:
            # No event loop running, create a new one
            loop = get_event_loop()
            return loop.run_until_complete(coro)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        params = self._get_actor_params() if self._is_actor else None
        rollout_name = self.config.rollout.name
        if self._is_rollout:
            if rollout_name == "vllm":
                inference_model = get_inference_model(self.rollout)

                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
                # For ServerAdapter, _engine might be None and needs async initialization
                if inference_model is None:
                    # Initialize the server adapter engine
                    print(f"[sync_rollout_weights] Initialize server adapter engine")
                    async def init_engine():
                        if hasattr(self.rollout, "_init_server_adapter"):
                            await self.rollout._init_server_adapter()
                        else:
                            print(f"[sync_rollout_weights] No _init_server_adapter method found")
                        return self.rollout._engine
                    
                    inference_model = self._run_async_safely(init_engine())
                    if inference_model is None:
                        raise RuntimeError(
                            f"Failed to initialize rollout engine. "
                            f"rollout type: {type(self.rollout)}, "
                            f"has _init_server_adapter: {hasattr(self.rollout, '_init_server_adapter')}"
                        )
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
        # loop = get_event_loop()
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
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    # loop.run_until_complete(self.update_weights(inference_model, [(key, tensor)]))
                    self._run_async_safely(self.update_weights(inference_model, [(key, tensor)]))
        get_torch_device().empty_cache()


    async def update_weights(self, inference_engine, params):
        # if self.rollout_device_mesh == None:
        #     infer_tp = self.config.rollout.tensor_model_parallel_size
        #     dp = torch.distributed.get_world_size() // infer_tp
        #     rollout_device_mesh = init_device_mesh(
        #         device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        #     )
        #     self.rollout_device_mesh = rollout_device_mesh
        print(f"[update_weights] rollout_device_mesh: {self.rollout_device_mesh}")
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        await sgl_update_weights(
            engine=inference_engine,
            params_batch=params,
            device_mesh_key="infer_tp",
            device_mesh=self.rollout_device_mesh,
        )

        if self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
            await inference_engine.flush_cache()

class DetachActorWorker(DetachNcclSync):
    def _get_actor_params(self):
        assert self._is_actor
        params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        return params

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


class DetachAsyncRolloutWorker(DetachNcclSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        ActorRolloutRefWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
