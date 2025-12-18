# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025 Huawei Ltd. and/or its affiliates
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
import time

import httpx
import torch
import torch.distributed
from checkpoint_engine.ps import ParameterServer, request_inference_to_update
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class CkptEngineWorker(Worker):
    def __init__(self, rank_offset, ps_world_size, inference_parallel_size):
        super().__init__()
        rank = self.rank + rank_offset
        self.ps_rank = rank
        self.ps_rank_offset = rank_offset
        self.ps_world_size = ps_world_size
        self.inference_parallel_size = inference_parallel_size
        self.ps = ParameterServer(rank=rank, world_size=ps_world_size)
        self.index = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_process_group(self):
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61020-61050"
        self.ps.init_process_group(device_index=0, master_port=60010)
        del os.environ["HCCL_NPU_SOCKET_PORT_RANGE"]

    def check_vllm_ready(self, uds: str | None = None):
        if self.ps_rank != self.ps_rank // self.inference_parallel_size * self.inference_parallel_size:
            return
        retry_num = 0
        transport = None
        if uds is not None:
            transport = httpx.HTTPTransport(uds=uds)
        while True:
            try:
                response = httpx.Client(transport=transport).get(f"{self.endpoint}/health", timeout=10)
                response.raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                retry_num += 1
                logger.warning(f"fail to check vllm ready, retry {retry_num} times, error: {e}")
                time.sleep(5)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def set_server_addresses(self, server_addresses: list[str]):
        # todo support multiple api server
        self.endpoint = f"http://{server_addresses[0]}"
        self.check_vllm_ready()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights_by_ckpt_engine(self):
        rank = self.rank
        src = rank // self.inference_parallel_size * self.inference_parallel_size

        def req_func(socket_paths: list[tuple[str, str]]) -> None:
            if rank == src:
                request_inference_to_update(
                    url=f"{self.endpoint}/collective_rpc",
                    socket_paths=dict(socket_paths),
                )

        checkpoint_name = f"sync_{self.index}"
        self.ps.register_checkpoint(checkpoint_name=checkpoint_name)
        self.ps.gather_metas(checkpoint_name)
        ranks = list(range(self.ps_rank_offset, self.ps_world_size))
        self.ps.update(checkpoint_name, req_func, ranks=ranks)
