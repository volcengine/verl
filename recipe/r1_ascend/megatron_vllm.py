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
import logging
import os

from omegaconf import DictConfig
from torch import nn

from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu, per_tensor_generator
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.profiler.performance import simple_timer
from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager as MVShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronVLLMShardingManager(MVShardingManager):
    def __init__(
        self,
        actor_module: nn.ModuleList,
        rollout,
        model_config: DictConfig,
        transformer_config,
        rollout_config: DictConfig,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        device_mesh,
        offload_param: bool = True,
        bridge=None,
    ):
        super().__init__(
            actor_module,
            rollout.inference_engine,
            model_config,
            transformer_config,
            rollout_config,
            layer_name_mapping,
            weight_converter,
            device_mesh,
            offload_param,
            bridge,
        )
        self.rollout = rollout

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            aggressive_empty_cache(force_sync=True)

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_megatron_model_to_gpu(self.actor_module, load_grad=False)

            if self.rollout_config.free_cache_engine:
                # NPU-ADAPTATION: Onload the model to NPU
                self.rollout.onload_model_weights()
                # NPU-ADAPTATION END
            if self.bridge is not None:
                per_tensor_param = self.bridge.export_weights(self.actor_module)
            else:
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
            model = self.model_runner.model
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(model)
            loaded_params = model.load_weights(per_tensor_param)

            # NPU-ADAPTATION:Perform special processing on the parameters of the MLA layer.
            if hasattr(model.model.layers[0].self_attn, "mla_attn"):
                self._process_mla()
            # NPU-ADAPTATION END

            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            logger.info(info)

            if self.offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            aggressive_empty_cache(force_sync=True)

            if self.rollout_config.free_cache_engine:
                # NPU-ADAPTATION: init kv caches
                self.rollout.init_cache_engine()
                # NPU-ADAPTATION END

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self.rollout_config.free_cache_engine:
            # NPU-ADAPTATION: free kv caches and offload model
            self.rollout.free_cache_engine()
            self.rollout.offload_model_weights()
            # NPU-ADAPTATION END
        for model in self.actor_module:
            model.train()

        aggressive_empty_cache(force_sync=True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    # NPU-ADAPTATION:Perform special processing on the parameters of the MLA layer.
    def _process_mla(self):
        for i in range(self.model_runner.model.model.start_layer, self.model_runner.model.model.end_layer):
            mla = self.model_runner.model.model.layers[i].self_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
            mla.process_weights_after_loading(None)
    # NPU-ADAPTATION END
