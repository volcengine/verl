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
"""ReMax config validation (v1 trainer) and the missing-baseline diagnostic (shared helper)."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.trainer_base import validate_remax_config


def _config(adv_estimator="remax", trainer_mode="sync", filter_enable=False):
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv_estimator, "filter_groups": {"enable": filter_enable}},
            "trainer": {"v1": {"trainer_mode": trainer_mode}},
        }
    )


def test_remax_sync_mode_is_accepted():
    validate_remax_config(_config())
    validate_remax_config(_config(adv_estimator="grpo", trainer_mode="colocate_async", filter_enable=True))


def test_remax_requires_sync_trainer_mode():
    with pytest.raises(NotImplementedError, match="trainer_mode=sync"):
        validate_remax_config(_config(trainer_mode="colocate_async"))


def test_remax_rejects_filter_groups():
    with pytest.raises(NotImplementedError, match="filter_groups"):
        validate_remax_config(_config(filter_enable=True))


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


class _FakeTQ:
    """In-memory stand-in for the TransferQueue calls used by ``_extract_remax_baselines``."""

    def __init__(self, rm_scores):
        self.rm_scores = rm_scores  # key -> list[float]
        self.put = {}
        self.cleared = []

    def kv_batch_get(self, keys, partition_id, select_fields):
        assert select_fields == ["rm_scores"]
        nested = torch.nested.as_nested_tensor([torch.tensor(self.rm_scores[k]) for k in keys], layout=torch.jagged)
        return {"rm_scores": nested}

    def kv_batch_put(self, keys, partition_id, fields):
        self.put.update(zip(keys, fields["reward_baselines"].tolist(), strict=True))

    def kv_clear(self, keys, partition_id):
        self.cleared.extend(keys)


def test_extract_remax_baselines_moves_greedy_session_into_reward_baselines(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base
    from verl.trainer.ppo.v1.utils import REMAX_BASELINE_SESSION_ID as BASE

    rm = {
        "a_0_0": [0.0, 0.5],
        "a_1_0": [0.0, 1.0],
        f"a_{BASE}_0": [0.0, 0.25],
        "b_0_0": [2.0],
        "b_1_0": [3.0],
        f"b_{BASE}_0": [0.75],
        f"b_{BASE}_1": [1.5],  # multi-output greedy session: the last output carries the outcome
    }
    fake = _FakeTQ(rm)
    monkeypatch.setattr(trainer_base, "tq", fake)
    keys = list(rm)
    batch = trainer_base.KVBatchMeta(partition_id="train", keys=keys, tags=[{"k": k} for k in keys])

    out = trainer_base.PPOTrainer._extract_remax_baselines(SimpleNamespace(), batch, metrics=(metrics := {}))

    assert out.keys == ["a_0_0", "a_1_0", "b_0_0", "b_1_0"]
    assert [t["k"] for t in out.tags] == out.keys
    assert fake.put == {"a_0_0": 0.25, "a_1_0": 0.25, "b_0_0": 1.5, "b_1_0": 1.5}
    assert sorted(fake.cleared) == sorted(k for k in keys if BASE in k)
    assert metrics["remax/baseline_reward_mean"] == pytest.approx((0.25 + 0.75 + 1.5) / 3)


def test_extract_remax_baselines_drops_prompt_without_greedy_session(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base
    from verl.trainer.ppo.v1.utils import REMAX_BASELINE_SESSION_ID as BASE

    rm = {"a_0_0": [1.0], "c_0_0": [2.0], f"c_{BASE}_0": [0.5]}
    fake = _FakeTQ(rm)
    monkeypatch.setattr(trainer_base, "tq", fake)
    batch = trainer_base.KVBatchMeta(partition_id="train", keys=list(rm), tags=[{}, {}, {}])
    out = trainer_base.PPOTrainer._extract_remax_baselines(SimpleNamespace(), batch, metrics=(metrics := {}))
    assert out.keys == ["c_0_0"]
    assert fake.put == {"c_0_0": 0.5}
    assert "a_0_0" in fake.cleared
    assert metrics["remax/baseline_missing_prompts"] == 1


def test_extract_remax_baselines_without_any_greedy_session_raises(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base

    rm = {"a_0_0": [1.0]}
    monkeypatch.setattr(trainer_base, "tq", _FakeTQ(rm))
    batch = trainer_base.KVBatchMeta(partition_id="train", keys=list(rm), tags=[{}])
    with pytest.raises(ValueError, match="no greedy baseline"):
        trainer_base.PPOTrainer._extract_remax_baselines(SimpleNamespace(), batch, metrics={})
