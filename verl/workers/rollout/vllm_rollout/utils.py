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
import hashlib
import logging
import os
import socket
from dataclasses import asdict
from functools import lru_cache
from types import MethodType
from typing import Callable, TypedDict

import torch
import zmq
from vllm.lora.request import LoRARequest

from verl.utils.device import get_torch_device, is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        # try to get ip from network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:  # noqa: BLE001
        # fallback to get ip from hostname
        logger.warning(f"fail to get ip from network interface, fallback to get ip from hostname: {e}")
        return socket.gethostbyname(socket.gethostname())


def npu_generate_uuid(rank: int) -> str:
    hash_str = hashlib.md5(str(f"{get_ip()}-{rank}").encode(), usedforsecurity=False).hexdigest()
    return "NPU-" + hash_str


def get_npu_real_device_id(pytorch_device_id=0):
    """
    Get the actual physical device ID corresponding to a PyTorch device ID.

    Args:
        pytorch_device_id: Internal PyTorch device ID (default: 0)

    Returns:
        Actual physical device ID

    Notes:
        Uses the ASCEND_RT_VISIBLE_DEVICES environment variable which
        works similarly to CUDA_VISIBLE_DEVICES for NPU devices.
        If the environment variable is not set, returns the input ID directly.
    """
    npu_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")

    if not npu_visible_devices:
        return pytorch_device_id

    devices = []
    for part in npu_visible_devices.split(","):
        part = part.strip()
        if part:
            try:
                devices.append(int(part))
            except ValueError:
                devices.append(part)

    assert pytorch_device_id < len(devices), (
        f"pytorch_device_id ({pytorch_device_id}) must less than len(devices) ({len(devices)})"
    )
    return devices[pytorch_device_id]


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
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
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
        global_rank = kwargs.get("rank", 0) + int(os.environ.get("VERL_VLLM_MULTIPROC_GLOBAL_RANK_OFFSET", "0"))
        local_rank = kwargs.get("local_rank", 0)
        kwargs["distributed_init_method"] = os.environ.get("DIST_INIT_METHOD", None)

        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()

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
        device = get_torch_device()

        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handle)
        buffer: torch.Tensor | None = None
        weights_to_load = []
        while True:
            payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = socket.recv_pyobj()
            if payload is None:
                # means the update is done
                process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
                device.synchronize()
                socket.send(b"")
                break
            if isinstance(payload, tuple):
                # an ipc handle that vLLM can use `func, args = handle`
                # and `func(*args)` to rebuild GPU tensor.
                buffer = rebuild_ipc(payload, self.device.index)
                assert buffer.dtype == torch.uint8
                socket.send(b"")
                continue
            assert isinstance(payload, list)
            assert buffer is not None
            weights = []
            for item in payload:
                shape = item["shape"]
                if isinstance(shape, list | tuple):
                    shape = torch.Size(shape)
                assert isinstance(shape, torch.Size)
                dtype, offset = item["dtype"], item["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                if not load:
                    tensor = tensor.clone()
                weights.append((item["name"], tensor))
            if load:
                self.model_runner.model.load_weights(weights=weights)
                del weights
                device.synchronize()
            else:
                weights_to_load.extend(weights)
            socket.send(b"")

        socket.close()
        del buffer
        gc.collect()
        device.empty_cache()
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
        logger.info(f"vLLM load weights, loaded_params: {len(lora_weights)}")
        del lora_weights
        gc.collect()
        get_torch_device().empty_cache()

    def report_device_id(self) -> str:
        """Report device ID for ZMQ handle."""
        from vllm.platforms import current_platform

        if not hasattr(self, "device_uuid") or not self.device_uuid:
            if is_npu_available:
                self.device_uuid = npu_generate_uuid(get_npu_real_device_id(self.device.index))
            else:
                self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid
