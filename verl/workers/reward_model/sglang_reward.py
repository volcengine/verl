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
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
# from verl.workers.config import 
from verl.workers.rollout.async_server import TokenOutput
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
from verl.workers.reward_model import BasePPORewardModel
from verl.workers.config import RewardModelConfig, HFModelConfig


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SGLangRewardModel(BasePPORewardModel):
    def __init__(
        self,
        config: RewardModelConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        actor_module = model_config.local_path
        trust_remote_code = model_config.trust_remote_code
        port = None
        kwargs = {}

        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        self._init_distributed_env(device_mesh_cpu=None, **kwargs)
        self._init_inference_engine(trust_remote_code, actor_module, port)

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # attention backend will be changed to fa3 if not specified
        attention_backend = engine_kwargs.pop("attention_backend", None)
        max_running_requests = self.config.get("max_num_seqs", None)

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            args = {
                "model_path": actor_module,
                "dtype": self.config.dtype,
                "mem_fraction_static": self.config.gpu_memory_utilization,
                "enable_memory_saver": True,
                "base_gpu_id": 0,
                "gpu_id_step": 1,
                "tp_size": self._tp_size,
                "node_rank": node_rank,
                "dist_init_addr": dist_init_addr,
                "nnodes": nnodes,
                "trust_remote_code": trust_remote_code,
                "max_running_requests": max_running_requests,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                "port": 30000 + rank,
                # NOTE(Chenyang): if you want to debug the SGLang engine output
                # please set the following parameters
                # Otherwise, it will make the engine run too slow
                # "log_level": "info",
                "log_level": "error",
                # log_requests=True,
                # log_requests_level=2,
                # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
                # max_running_requests=1,
                "mm_attention_backend": "fa3",
                "attention_backend": attention_backend if attention_backend is not None else "fa3",
                # In async mode, we want token in token out.
                "skip_tokenizer_init": True,
                "is_embedding": True,
            }

            # add server specific args
            args["first_rank_in_node"] = first_rank_in_node
            args["timeout"] = self.config.server["timeout"]
            args["max_attempts"] = self.config.server["max_attempts"]
            args["retry_delay"] = self.config.server["retry_delay"]
            args["max_connections"] = self.config.server["max_connections"]
            args["max_start_wait_time"] = self.config.server["max_start_wait_time"]
            self._engine = AsyncHttpServerAdapter(**args)

            # from sglang.srt.server_args import ServerArgs
            # from sglang.srt.entrypoints.http_server import launch_server
            # kwargs = {'model_path': 'Qwen/Qwen2.5-1.5B-Instruct', 'dtype': 'bfloat16', 'mem_fraction_static': 0.8, 'enable_memory_saver': True, 'base_gpu_id': 0, 'gpu_id_step': 1, 'tp_size': 2, 'node_rank': 0, 'dist_init_addr': None, 'nnodes': 1, 'trust_remote_code': False, 'max_running_requests': 1024, 'port': 30002, 'log_level': 'info', 'mm_attention_backend': 'fa3', 'attention_backend': 'fa3', 'skip_tokenizer_init': True, 'is_embedding': True}
            # server_args = ServerArgs(**kwargs)
            # launch_server(server_args)
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    def compute_reward(self, data: DataProto):
        pass

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])
