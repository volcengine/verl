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
import asyncio
import functools
import importlib
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generator, Optional

import torch
from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

__all__ = ["BaseRollout"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class BaseRollout(ABC):
    """Base class for rollout."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        pass

    @abstractmethod
    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Args:
            prompts: The input prompts.

        Returns:
            The output sequences.
        """
        raise NotImplementedError


_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}


def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:
    """Get the rollout class by name.

    Args:
        rollout_name: The name of the rollout.
        mode: The mode of the rollout, async: server mode.

    Returns:
        The rollout class.
    """
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)


class TokenOutput(BaseModel):
    token_ids: list[int]
    """response token ids"""
    log_probs: Optional[list[float]] = None
    """logprobs of response token ids"""
    routed_experts: Optional[Any] = None
    """routed experts of response token ids"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    start_model_version: Optional[int] = None
    """model version when the request is started"""
    finish_model_version: Optional[int] = None
    """model version when the request is finished"""


def resume_on_abort(func):
    """Automatically resume generation when the rollout is interrupted for updating weights,
    this can make AgentLoop agnostic to the rollout interruption.
    """

    @functools.wraps(func)
    async def wrapper(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs,
    ):
        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            routed_experts=[],
            num_preempted=0,
            start_model_version=self.model_version,
        )

        retry_count = 0
        while True:
            # 1. wait for weight finish
            await self.resume_event.wait()

            # 2. continue generate
            output: TokenOutput = await func(
                self,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=deepcopy(sampling_params),
                request_id=f"{request_id}_{retry_count}",
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )

            # 3. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            if output.routed_experts is not None:
                final_output.routed_experts.extend(output.routed_experts)
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # 4. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            if output.stop_reason != "abort":
                break
            retry_count += 1
            logger.info(f"Resume generation for request {request_id} after abort, retry count: {retry_count}")

        final_output.finish_model_version = self.model_version
        return final_output

    return wrapper


class BaseRolloutServer(ABC):
    """Base class for rollout server.

    Args:
        model_version: initial model version of the rollout server.
    """

    def __init__(self, model_version: int = 0):
        self.resume_event = asyncio.Event()
        self.resume_event.set()
        self.model_version = model_version

    @resume_on_abort
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        raise NotImplementedError

    @abstractmethod
    async def abort_all_requests(self):
        """Abort all requests with partial rollout."""
        self.resume_event.clear()

    async def resume_all_requests(self):
        """Resume all requests with partial rollout."""
        self.model_version += 1
        self.resume_event.set()

    @abstractmethod
    async def start_profile(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def stop_profile(self, **kwargs):
        raise NotImplementedError
