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
"""
The base class for reward model
"""

import importlib
from abc import ABC, abstractmethod

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.workers.config import HFModelConfig, RewardModelConfig

__all__ = ["BaseRewardModel"]


class BaseRewardModel(ABC):
    """base class for reward model"""

    def __init__(
        self,
        config: RewardModelConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """Resume reward model weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        pass

    @abstractmethod
    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        """Computing reward given input_ids. The transformers should output a tensor with shape
           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: a data pass protocol containing "reward". Only the [EOS] position contains the reward.
            Other position should have zero reward. Note that this may change in the future if we use
            dense reward. So, we leave the interface for general case.
            - reward: [batch_size, sequence_length].

        """
        pass


_REWARD_MODEL_REGISTRY = {
    "sglang": "verl.workers.roles.reward_model_engine.sglang_reward_model.SGLangRewardModel",
}


def get_reward_model_class(reward_model_name: str) -> type[BaseRewardModel]:
    """Get the reward model class by name.

    Args:
        reward_model_name: The name of the reward model.

    Returns:
        The reward model class.
    """
    assert reward_model_name in _REWARD_MODEL_REGISTRY, f"Reward Model {reward_model_name} with mode not found"
    fqdn = _REWARD_MODEL_REGISTRY[reward_model_name]
    module_name, class_name = fqdn.rsplit(".", 1)
    reward_model_module = importlib.import_module(module_name)
    return getattr(reward_model_module, class_name)
