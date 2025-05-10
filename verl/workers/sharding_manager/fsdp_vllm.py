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

import os
import inspect
import logging
import time
from typing import List
import torch
import numpy as np
from packaging import version
from peft import PeftModel
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh
from collections import OrderedDict

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import log_gpu_memory_usage, log_print
from verl.third_party.vllm import vllm_version
from vllm.version import __version__ as VLLM_VERSION

from .base import BaseShardingManager
from .patch import patched_ds_v3_load_weights

from peft.tuners.tuners_utils import BaseTunerLayer
import torch.distributed as dist
from peft.utils.save_and_load import get_peft_model_state_dict
from peft.utils.other import transpose

from vllm.engine.llm_engine import LLMEngine
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.models import LRUCacheLoRAModelManager, LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.peft_helper import PEFTHelper
from dataclasses import dataclass, asdict
from msgspec import Struct, field


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

class TensorLoRARequest(LoRARequest):
    peft_config:dict = field(default=None)
    lora_tensors:dict = field(default=None)

class VLLMHijack():
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors
            """
            try:
                supported_lora_modules = (
                    self._adapter_manager.supported_lora_modules)
                packed_modules_mapping = (
                    self._adapter_manager.packed_modules_mapping)
                expected_lora_modules: List[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(
                            packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if (hasattr(model, "hf_to_vllm_mapper")
                        and model.hf_to_vllm_mapper is not None):
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper
                )
            except Exception as e:
                raise e

            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(f"LoRA added vocab size {lora.extra_vocab_size} "
                                f"is greater than lora_extra_vocab_size "
                                f"{self.lora_config.lora_extra_vocab_size}.")
            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            log_print(f"SimonDbg: in monkey patch do_hijack {target_cls=} {target_method_name=}")
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)

class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None,
                 load_format: str = 'dummy_hf',
                ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = 'dummy' not in load_format
        VLLMHijack.hijack()

    def __enter__(self):
        torch.cuda.empty_cache()
        tac = time.time()
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        peft_config = None
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            peft_config = self.module._fsdp_wrapped_module.peft_config.get('default', None)
            # log_print(f"{peft_config=}")
            lora_params = OrderedDict()
            if isinstance(self.module, FSDP):
                log_print(f"SimonDbg: <1>. PeftModel with PSDF")
                with FSDP.summon_full_params(self.module, writeback=False):
                    if self.base_sync_done:
                        lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                        lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu() 
                                       for name, param in lora_params.items()}
                    else:
                        model = self.module._fsdp_wrapped_module.base_model.model.to('cpu')
                        for name, param in model.state_dict().items():
                            if any(x in name for x in ['_flat_param', 'lora_']):
                                continue
                            name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                            lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, 'full_tensor') else param.detach().cpu()
                torch.cuda.empty_cache()
            else:
                log_print(f"SimonDbg: <2>. PeftModel without PSDF")
                lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                else:
                    model = self.module._fsdp_wrapped_module.base_model.model.to('cpu')
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ['_flat_param', 'lora_']):
                            continue
                        name = name.replace("_fsdp_wrapped_module.","").replace(".base_layer","")
                        lora_params[name] = param.detach().cpu()
            params = lora_params
        elif isinstance(self.module, FSDP):
            log_print(f"SimonDbg: <3>. Not PeftModel with FSDP")
            with FSDP.summon_full_params(self.module, writeback=False):
                # params = self.module._fsdp_wrapped_module.base_model.state_dict()
                params = self.module.state_dict()
        else:
            log_print(f"SimonDbg: <4>. Not PeftModel without FSDP")
            params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)

        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'

        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params, peft_config=peft_config)
            log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
            del params
            torch.cuda.empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

        tic = time.time()
        log_print(f"FSDPVLLMShardingManager: __enter__ model weight syncing time cost: {tic-tac:.2f}s")

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        torch.cuda.empty_cache()
        # if load_format is dummy and lora-fsdp, base model must be synced, actor_model was offloading to avoid OOM, need load back to gpu
        self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
            # print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config = None):
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        if peft_config:
            if self.base_sync_done:
                log_print(f"SimonDbg: update_params with LoRA") 
                lora_int_id=int(time.time_ns() // 1_000_000)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                if self.tp_rank == 0:
                    self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                return
            else:
                log_print(f"SimonDbg: update base-model weights")
                def replace_lora_wrapper(k):
                    stacked_params = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k
                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        if model.config.architectures[0] in ['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM']:
            loaded_params = patched_ds_v3_load_weights(
                model, ((name, param.full_tensor() if hasattr(param, 'full_tensor') else param)
                        for name, param in updated_params.items()))
        else:
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if hasattr(param, 'full_tensor') else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")
