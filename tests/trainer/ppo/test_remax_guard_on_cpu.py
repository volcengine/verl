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
"""ReMax needs a greedy baseline rollout that only the v0 trainer produces; fail clearly otherwise."""

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def test_v1_trainer_rejects_remax_at_startup():
    config = OmegaConf.create({"algorithm": {"adv_estimator": "remax"}})

    class _Trainer(PPOTrainer):
        pass

    _Trainer.__abstractmethods__ = frozenset()  # only __init__ is under test
    with pytest.raises(NotImplementedError, match="trainer.use_v1=False"):
        _Trainer(config)


def test_compute_advantage_remax_without_baselines_raises_clear_error():
    batch = TensorDict(
        {
            "token_level_rewards": torch.zeros(2, 3),
            "response_mask": torch.ones(2, 3),
        },
        batch_size=2,
    )
    data = DataProto(batch=batch)
    with pytest.raises(ValueError, match="reward_baselines"):
        compute_advantage(data, adv_estimator="remax")
