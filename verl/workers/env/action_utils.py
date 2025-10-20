# Copyright 2024 Bytedance Ltd. and/or its affiliates

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from verl.envs.libero.utils import invert_gripper_action, normalize_gripper_action


def prepare_actions_simplevla_libero(
    raw_chunk_actions,
) -> torch.Tensor:
    normalized_action = normalize_gripper_action(raw_chunk_actions, binarize=True)
    inverted_action = invert_gripper_action(normalized_action)
    return inverted_action


def prepare_actions(
    simulator_type,
    raw_chunk_actions,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
) -> torch.Tensor:
    if simulator_type == "libero":
        chunk_actions = prepare_actions_simplevla_libero(
            raw_chunk_actions=raw_chunk_actions,
        )
    else:
        raise NotImplementedError

    return chunk_actions
