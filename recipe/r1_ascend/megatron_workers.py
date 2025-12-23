# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
# Adapted from https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py
"""
The main entry point to run the PPO algorithm
"""

import torch
from mindspeed.core.megatron_basic.requirements_basic import dummy_compile

# NPU-ADAPTATION: Save the original and dummy copies of `torch.compile`.
from mindspeed.patch_utils import MindSpeedPatchesManager
from omegaconf import DictConfig

from verl.workers.megatron_workers import ActorRolloutRefWorker as ARRWorker
from verl.workers.rollout import base

MindSpeedPatchesManager.patches_info["torch.compile"].remove_patch()
TRUE_COMPILE = torch.compile
DUMMY_COMPILE = dummy_compile
# NPU-ADAPTATION END


base._ROLLOUT_REGISTRY[("vllm", "sync")] = "recipe.r1_ascend.vllm_rollout_spmd.vLLMRollout"


class ActorRolloutRefWorker(ARRWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role)

    def _build_rollout(self, *args, **kwargs):
        """
        Build the rollout with temporary reversion to true torch.compile.
        """
        # Temporarily restore true torch.compile for the rollout build
        torch.compile = TRUE_COMPILE

        # Call parent method with original torch.compile
        super()._build_rollout(*args, **kwargs)

        # Revert to dummy_compile after rollout is built
        torch.compile = DUMMY_COMPILE
