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

import gc
import logging
import os
from dataclasses import asdict
from types import MethodType
from typing import Callable, TypedDict

import torch
import zmq
from vllm.lora.request import LoRARequest

from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.

    Reference: https://github.com/vllm-project/vllm/blob/8a297115e2367d463b781adb86b55ac740594cf6/vllm/config/lora.py#L27
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)

# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer

class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int

class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """
    def __new__(cls, **kwargs):
        kwargs["rank"] += int(os.environ.get("VERL_VLLM_MULTIPROC_GLOBAL_RANK_OFFSET", "0"))
        kwargs["distributed_init_method"] = os.environ.get("DIST_INIT_METHOD", None)
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            # Apply vllm fp8 patches
            # Will remove the patch after vllm support on-the-fly quant for rollout natively.
            apply_vllm_fp8_patches()
        return super().__new__(cls)

    def monkey_patch_compute_logits(self, vocab_size: int):
        _monkey_patch_compute_logits(self.model_runner.model, vocab_size)
    
    def _fetch_weights(self, zmq_handle: str, load: bool = True):
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handle)
        weights_to_load = []
        while True:
            payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = (
                socket.recv_pyobj()
            )
            if payload is None:
                # means the update is done
                process_weights_after_loading(
                    self.model_runner.model, self.model_config, self.device
                )
                torch.cuda.synchronize()
                socket.send(b"")
                break
            tensor = rebuild_ipc(payload, self.device.index)
            socket.send(b"")

            # Get the next metadata containing tensor name
            metadata: dict | None = socket.recv_pyobj()
            if metadata is None:
                del tensor
                continue

            name = metadata["name"]
            weights = [(name, tensor)]
            if load:
                self.model_runner.model.load_weights(weights=weights)
                del weights
                torch.cuda.synchronize()
            else:
                weights_to_load.extend(weights)
            socket.send(b"")

        socket.close()
        gc.collect()
        torch.cuda.empty_cache()
        return weights_to_load

    def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
        model_runner = self.model_runner
        model = model_runner.model
        patch_vllm_moe_model_weight_loader(model)

        # Add the FP8 related logic here as sharding manager has been deprecated.
        # Check if FP8 quantization is enabled and apply appropriate weight loading
        vllm_config = model_runner.vllm_config
        if is_fp8_model(model_runner.vllm_config):
            logger.info(f"FP8 model detected (async): {vllm_config.quant_config}")
            # Convert bf16 weights to fp8 format before loading
            weights = self._fetch_weights(zmq_handles[self.report_device_id()], load=False)
            loaded_params = load_quanted_weights(weights, model_runner)
            logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
        else:
            logger.info("Loading standard weights (non-FP8, async)")
            weights = self._fetch_weights(zmq_handles[self.report_device_id()], load=True)

    def update_lora_weights_from_ipc(self, peft_config: dict, zmq_handles: dict[str, str]):
        # In async mode, make sure the old lora is removed before adding the new one
        self.remove_lora(VLLM_LORA_INT_ID)
        lora_weights = self._fetch_weights(zmq_handles[self.report_device_id()], load=False)
        lora_request = LoRARequest(
            lora_name=VLLM_LORA_NAME,
            lora_int_id=VLLM_LORA_INT_ID,
            lora_path=VLLM_LORA_PATH,
            peft_config=asdict(peft_config),
            lora_tensors=dict(lora_weights),
        )
        self.add_lora(lora_request)
        del lora_weights
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"vLLM load weights, loaded_params: {len(lora_weights)}")

    def report_device_id(self) -> str:
        """Report device ID for ZMQ handle."""
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid
