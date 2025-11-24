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


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def update_weights_per_tensor_from_ipc(self, zmq_handles: dict[str, str]):
        """Update weights per tensor from IPC handles."""
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from verl.utils.vllm.utils import rebuild_ipc
        import gc
        import zmq

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handles[self.report_device_id()])

        while True:
            # Receive either a tensor handle (tuple) or None (end signal)
            payload: tuple[Callable, tuple] | None = socket.recv_pyobj()
            if payload is None:
                # means the update is done
                process_weights_after_loading(
                    self.model_runner.model, self.model_config, self.device
                )
                torch.cuda.synchronize()
                socket.send(b"")
                break

            # Rebuild the tensor from IPC handle
            tensor = rebuild_ipc(payload, self.device.index)
            socket.send(b"")

            # Get the next metadata containing tensor name
            metadata: dict | None = socket.recv_pyobj()
            if metadata is None:
                del tensor
                continue

            # Load this single tensor
            name = metadata["name"]
            weights = [(name, tensor)]
            self.model_runner.model.load_weights(weights=weights)
            del weights
            torch.cuda.synchronize()
            socket.send(b"")

        socket.close()
        gc.collect()
        torch.cuda.empty_cache()

    def update_lora_weights_per_tensor_from_ipc(self, peft_config: dict, zmq_handles: dict[str, str]):
        """Update LoRA weights per tensor from IPC handles."""
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from verl.utils.vllm.utils import rebuild_ipc
        from dataclasses import asdict
        from vllm.lora.request import LoRARequest
        import gc
        import zmq

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handles[self.report_device_id()])

        # In async mode, make sure the old lora is removed before adding the new one
        self.remove_lora(VLLM_LORA_INT_ID)
        lora_weights = []
        while True:
            # Receive either a tensor handle (tuple) or None (end signal)
            payload: tuple[Callable, tuple] | None = socket.recv_pyobj()
            if payload is None:
                # means the update is done
                torch.cuda.synchronize()
                socket.send(b"")
                break

            # Rebuild the tensor from IPC handle
            tensor = rebuild_ipc(payload, self.device.index)
            socket.send(b"")

            # Get the next metadata containing tensor name
            metadata: dict | None = socket.recv_pyobj()
            if metadata is None:
                # Should not happen in per-tensor mode, but handle gracefully
                del tensor
                continue

            name = metadata["name"]
            lora_weights.append((name, tensor))
            torch.cuda.synchronize()
            socket.send(b"")

        lora_request = LoRARequest(
            lora_name=VLLM_LORA_NAME,
            lora_int_id=VLLM_LORA_INT_ID,
            lora_path=VLLM_LORA_PATH,
            peft_config=asdict(peft_config),
            lora_tensors=dict(lora_weights),
        )
        self.add_lora(lora_request)

        socket.close()
        gc.collect()
        torch.cuda.empty_cache()

    def report_device_id(self) -> str:
        """Report device ID for ZMQ handle."""
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
