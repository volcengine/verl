# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
import vllm.distributed.parallel_state as vllm_ps

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, collect_all_to_all, dispatch_dp_compute_data_proto, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class DynRLActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported in dynamic parallel mode.")

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout

            from .fsdp_vllm import FSDPVLLMDynShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
            # lora_kwargs = {}
            if vllm_mode == "customized":
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            elif vllm_mode == "spmd":
                from .vllm_dyn_rollout_spmd import vLLMDynRollout

                vllm_rollout_cls = vLLMDynRollout
                rollout = vllm_rollout_cls(model_path=local_path, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, device_mesh=rollout_device_mesh, trust_remote_code=trust_remote_code, **lora_kwargs)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMDynShardingManager(
                module=self.actor_module_fsdp,
                inference_engine_dict=rollout.candidated_inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name in ["sglang", "sglang_async"]:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported in dynamic parallel mode.")

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode={"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_all_to_all})
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())

        assert self._is_rollout

        log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        if vllm_ps.get_tensor_model_parallel_rank() == 0:
            self.rollout.generate_sequences(prompts=prompts)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def postprocess_generate_sequences(self, gen_batch_output: DataProto):
        # Support all hardwares
        gen_batch_output = gen_batch_output.to(get_torch_device().current_device())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        gen_batch_output.meta_info.update(meta_info)

        output = self.rollout.postprocess_generate_sequences(gen_batch_output)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wake_up(self):
        self.rollout_sharding_manager.__enter__()
        # TODO(lkm): return timing_s/reshard
        print(f"rank:{torch.distributed.get_rank()}, timing_s/reshard: {self.rollout_sharding_manager.timing['reshard']}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sleep(self):
        self.rollout.reset_inference_engine()
        self.rollout_sharding_manager.__exit__(None, None, None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def setup_buffer(self, buffer_addr: str):
        return self.rollout.setup_buffer(buffer_addr)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_num_unfinished_requests(self):
        return self.rollout.get_num_unfinished_requests()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def scale_up(self):
        await self.rollout.scale_up()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def execute_method(self, method, *args, **kwargs):
        return self.rollout.execute_method(method, *args, **kwargs)

    def execute_model(self, *args, **kwargs):
        return self.rollout.execute_model(*args, **kwargs)
