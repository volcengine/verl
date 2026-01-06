# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .llm_router import LLMRouter
from .notify import notify_llm_server_address, wait_for_server


class LightningRayTrainer(RayPPOTrainer):
    def init_workers(self):
        super().init_workers()
        assert self.async_rollout_manager is not None, "LightningRayTrainer only works with AgentLoop"
        assert self.config.get("lightning_trainer") is not None, "config.lightning_trainer is required"

        self.llm_router = LLMRouter.options(
            name="LLMRouter",  # name required for ray.get_actor later
        ).remote(
            config=self.config,
            tokenizer=self.tokenizer,
            server_handles=self.async_rollout_manager.server_handles,
        )

        router_addr = ray.get(self.llm_router.get_server_address.remote())
        notify_llm_server_address(router_addr)

        ## wait for agent server ready before training
        if self.config.lightning_trainer.health_check_url:
            wait_for_server(
                self.config.lightning_trainer.agent_server_addr,
                self.config.lightning_trainer.health_check_url,
                timeout=self.config.lightning_trainer.health_check_timeout,
            )
