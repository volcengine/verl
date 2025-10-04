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

"""FlowRL-specific configuration extensions."""

from dataclasses import dataclass, field
from typing import Optional

from verl.trainer.config import AlgoConfig


@dataclass
class FlowRLAlgoConfig(AlgoConfig):
    """Extended algorithm configuration for FlowRL.

    Adds FlowRL-specific parameters to the base AlgoConfig.

    Args:
        tb_coef (float): Trajectory balance coefficient for FlowRL.
            Controls the temperature of the flow matching objective.
            Default: 15.0 (as used in FlowRL paper)
    """

    tb_coef: float = 15.0


def add_flowrl_config_to_omegaconf(config):
    """Add FlowRL-specific configuration parameters to an existing config.

    This function modifies the config in-place to add FlowRL parameters
    if they don't already exist.

    Args:
        config: OmegaConf DictConfig object to modify

    Returns:
        Modified config object with FlowRL parameters added
    """
    from omegaconf import OmegaConf

    # Add tb_coef to algorithm config if not present
    if not hasattr(config.algorithm, 'tb_coef'):
        config.algorithm.tb_coef = 15.0

    # Add proj_layer to actor config if not present
    if hasattr(config, 'actor_rollout_ref') and hasattr(config.actor_rollout_ref, 'actor'):
        if not hasattr(config.actor_rollout_ref.actor, 'proj_layer'):
            config.actor_rollout_ref.actor.proj_layer = 3

        if not hasattr(config.actor_rollout_ref.actor, 'proj_dropout'):
            config.actor_rollout_ref.actor.proj_dropout = 0.1

    return config
