# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging

import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse

from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class AsyncSglangServer(AsyncServerBase):
    def __init__(self, config: DictConfig, dp_size: int, dp_rank: int, wg_prefix: str):
        super().__init__()
        self.config = config
        rollout_config = config.get("rollout", {})
        self._tp_size = rollout_config.get("tensor_model_parallel_size", 1)
        self._dp_size = dp_size
        self._dp_rank = dp_rank
        self.wg_prefix = wg_prefix
        self.workers = []

    async def init_engine(self):
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        matched_actors = [actor for actor in all_actors if actor.get("name", None).startswith(self.wg_prefix + "WorkerDict_")]

        # TODO support multi node
        for matched_actor in matched_actors:
            current_rank = int(matched_actor["name"].split(":")[-1])

            # send to all works in this tp group, because sglang is SPMD
            if current_rank >= self._dp_rank * self._tp_size and current_rank < (self._dp_rank + 1) * self._tp_size:
                self.workers.append(ray.get_actor(**matched_actor))

    async def chat_completion(self, raw_request: Request):
        request = await raw_request.json()

        output_dp_lst = []
        for worker in self.workers:
            output_future = worker.execute_method.remote("chat_completion", request)
            output_dp_lst.append(output_future)
        outputs = await asyncio.gather(*output_dp_lst)

        for output in outputs:
            if output is not None:
                return JSONResponse(output)
        raise RuntimeError("AsyncSglangServer No output from workers self._dp_rank: {self._dp_rank}, self._tp_size: {self._tp_size}, self.workers: {self.workers}")

    async def wake_up(self):
        for worker in self.workers:
            worker.resume.remote()

    async def sleep(self):
        for worker in self.workers:
            worker.offload.remote()
