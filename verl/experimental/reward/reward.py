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
import logging
import os

import ray
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    def __init__(self, config: DictConfig, reward_model_handle: ray.actor.ActorHandle = None):
        self.config = config
        self.reward_model_handle = reward_model_handle
        self._init_reward_fn()

    def _init_reward_fn(self):
        assert self.config.reward_model.reward_manager == "dapo", "Only DAPORewardFunction is supported now."
        from .reward_loop import DAPORewardLoop

        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)

        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        self.compute_score = get_custom_reward_fn(self.config)
        self.reward_fn = DAPORewardLoop(
            self.config, self.input_tokenizer, self.compute_score, self.reward_model_handle, self.reward_model_tokenizer
        )

    async def compute_score(self, data: DataProto) -> DataProto:
        return await self.reward_fn.compute_score(data)
