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

import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
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
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_ipv6
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.roles.reward_model_engine.base import BaseRewardModel
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


def _post_process_outputs(output, is_embedding):
    if is_embedding:
        scores = [o["embedding"][-1] for o in output]
        return scores
    else:
        output_ids = [o["output_ids"] for o in output]
        return output_ids


class SGLangRewardModel(BaseRewardModel):
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

        reward_type = self.config.model_type
        if reward_type == "discriminative":
            self.is_embedding = True
        elif reward_type == "generative":
            self.is_embedding = False
            self._init_sampling_params()
        else:
            raise ValueError(f"reward type {reward_type} not supported")

        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        self._init_distributed_env(device_mesh_cpu=None)
        self._init_inference_engine(trust_remote_code, actor_module, port)

    def _init_distributed_env(self, device_mesh_cpu):
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
                "log_level": "info",
                # "log_level": "error",
                # log_requests=True,
                # log_requests_level=2,
                # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
                # max_running_requests=1,
                "mm_attention_backend": "fa3",
                "attention_backend": attention_backend if attention_backend is not None else "fa3",
                # In async mode, we want token in token out.
                "skip_tokenizer_init": True,
                # For embedding models
                "is_embedding": self.is_embedding,
                # server specific args
                "timeout": self.config.server_config["timeout"],
                "max_attempts": self.config.server_config["max_attempts"],
                "retry_delay": self.config.server_config["retry_delay"],
                "max_connections": self.config.server_config["max_connections"],
                "max_start_wait_time": self.config.server_config["max_start_wait_time"],
                "first_rank_in_node": first_rank_in_node,
            }
            self._engine = AsyncHttpServerAdapter(**args)
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.max_new_tokens,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)) or "stop" in str(k):
                kwargs[k] = self.config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        self.sampling_params = kwargs

    def compute_reward(self, rm_input_ids):
        if self._tp_rank == 0:
            loop = asyncio.new_event_loop()
            if self.is_embedding:
                output = loop.run_until_complete(self._engine.async_reward_score(prompt=None, input_ids=rm_input_ids))
            else:
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None, sampling_params=self.sampling_params, input_ids=rm_input_ids
                    )
                )
        else:
            output = None

        dist.barrier()
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )

        output = _post_process_outputs(output, self.is_embedding)
        return output

    async def resume(self, tags: list[str]):
        """Resume reward model weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])
