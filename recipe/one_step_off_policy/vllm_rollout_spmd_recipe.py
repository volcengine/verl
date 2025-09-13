# recipe/one_step_off_policy/vllm_rollout_spmd_recipe.py
from verl.workers.rollout.vllm_rollout import (
    vLLMRollout as _BaseSync,
    vLLMAsyncRollout as _BaseAsync,
)

import asyncio
import getpass
import logging
import os
import pickle
import socket
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import DictConfig, ListConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class RecipevLLMAsyncRollout(_BaseAsync):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

        # weight staging (last-write-wins)
        self._lock = asyncio.Lock()
        self._pending = None
        self._applied_version = -1

        self.address = self._init_zeromq()
    
    def execute_method(self, method: str | bytes, *args, **kwargs):
        """Client-side call: send REQ to our server (_loop_forever) and wait for reply."""

        try:
            asyncio.get_running_loop()
            raise RuntimeError("Calling execute_method in the same event-loop can lead to deadlock - use asyncio.to_thread(...) to avoid blocking.")
        except RuntimeError:
            logger.warning("Calling execute_method in the same event-loop can lead to deadlock - use asyncio.to_thread(...) to avoid blocking.")
            pass

        ctx = zmq.Context.instance()
        with ctx.socket(zmq.REQ) as s:
            s.connect(self.address)  # set by _init_zeromq()
            s.send(pickle.dumps((method, args, kwargs)))
            reply = s.recv()
        return pickle.loads(reply)

    async def _apply_if_any(self):
        async with self._lock:
            item = self._pending
            self._pending = None
        if not item:
            return
        version, pairs = item

        model = self.inference_engine.worker.model_runner.model

        # TODO mirror sync path if you patch MoE loader
        # from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
        # patch_vllm_moe_model_weight_loader(model)

        model.load_weights(pairs)
        self._applied_version = version

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            ret = self._load_model(*args, **kwargs)
            await self._apply_if_any()
            return ret
        elif method == "sleep":
            return await self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return await self.wake_up(*args, **kwargs)
        elif method == "load_weights_from_tensors":
            # Expect: kwargs["pairs"] = [(name: str, cpu_tensor), ...]
            pairs = kwargs.get("pairs", [])
            assert pairs != [], "weights pairs detected to be None when calling load_weights_from_tensors. This should not happen and can lead to weights updating failure."
            async with self._lock:
                # last-write-wins; bump version monotonically per-process
                self._pending = (self._applied_version + 1, pairs)
            if self.inference_engine is not None:
                await self._apply_if_any()
            return True
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
