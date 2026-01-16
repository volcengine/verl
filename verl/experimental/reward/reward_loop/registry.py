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

from typing import Callable

from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase

__all__ = ["register", "get_reward_loop_manager_cls"]

REWARD_LOOP_MANAGER_REGISTRY: dict[str, type[RewardLoopManagerBase]] = {}


def register(name: str) -> Callable[[type[RewardLoopManagerBase]], type[RewardLoopManagerBase]]:
    """Decorator to register a reward loop manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop manager.
    """

    def decorator(cls: type[RewardLoopManagerBase]) -> type[RewardLoopManagerBase]:
        if name in REWARD_LOOP_MANAGER_REGISTRY and REWARD_LOOP_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"reward loop manager {name} has already been registered: {REWARD_LOOP_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_LOOP_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_loop_manager_cls(name: str) -> type[RewardLoopManagerBase]:
    """Get the reward loop manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop manager.

    Returns:
        `(type)`: The reward loop manager class.
    """
    if name not in REWARD_LOOP_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward loop manager: {name}")
    return REWARD_LOOP_MANAGER_REGISTRY[name]
