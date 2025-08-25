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

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.rollout.rollout_server import RolloutMode, RolloutServer
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

logger = logging.getLogger(__file__)


_rollout_worker_actor_cls = ray.remote(SGLangRollout)


class SGLangRolloutServer(RolloutServer):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            actor_module=self.model_config.local_path,
            config=self.config,
            processing_class=self.model_config.get_processor(),
            model_hf_config=self.model_config.hf_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        return worker_dict_cls

    async def init_server(self):
        """Init rollout server."""
        self._server_address = await self.workers[0].get_server_address.remote()
        self._server_handle = self.workers[0]

    async def wake_up(self):
        """Wake up rollout server."""
        assert self.mode != RolloutMode.STANDALONE, "wake_up shoud not be called in standalone mode"
        await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])

    async def sleep(self):
        """Sleep rollout server."""
        assert self.mode != RolloutMode.STANDALONE, "sleep shoud not be called in standalone mode"
        await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])

    async def update_weights(self):
        """Update weights of standalone rollout server."""
        assert self.mode == RolloutMode.STANDALONE, "update_weights should be called in standalone mode"
        raise NotImplementedError
