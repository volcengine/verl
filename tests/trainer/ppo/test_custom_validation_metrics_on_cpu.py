# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.metric_utils import CustomValidationMetricsFn, load_custom_validation_metrics_fn
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

_METRICS_MODULE = """
LAST_CALL = {}


def difficulty_metrics(data_sources, reward_extra_infos):
    LAST_CALL["data_sources"] = data_sources
    LAST_CALL["reward_extra_infos"] = reward_extra_infos
    accs = {}
    for difficulty, acc in zip(reward_extra_infos["difficulty"], reward_extra_infos["acc"], strict=True):
        accs.setdefault(difficulty, []).append(acc)
    return {f"val-aux/difficulty/{key}/acc/mean": sum(vals) / len(vals) for key, vals in accs.items()}


def not_a_dict(data_sources, reward_extra_infos):
    return [1.0]


def builtin_collision(data_sources, reward_extra_infos):
    return {"val-aux/num_turns/mean": 0.0}


NOT_CALLABLE = 1
"""

DATA_SOURCES = ["math", "math", "code", "code"]
SAMPLE_UIDS = ["u0", "u0", "u1", "u1"]
SAMPLE_TURNS = [1, 2, 3, 4]

ValMetricsUpdate = Callable[[CustomValidationMetricsFn | None, dict[str, list[Any]]], dict[str, Any]]


class _StubV1Trainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


@pytest.fixture
def metrics_module(tmp_path: Path) -> str:
    module_path = tmp_path / "my_metrics.py"
    module_path.write_text(_METRICS_MODULE)
    return str(module_path)


def _load(path: str | None, name: str | None) -> CustomValidationMetricsFn | None:
    return load_custom_validation_metrics_fn(
        OmegaConf.create({"custom_validation_metrics": {"path": path, "name": name}})
    )


def _reward_extra_infos() -> dict[str, list[Any]]:
    return {
        "reward": [1.0, 0.0, 1.0, 1.0],
        "acc": [1.0, 0.0, 1.0, 1.0],
        "difficulty": ["easy", "hard", "hard", "hard"],
    }


def _legacy_val_metrics(
    custom_fn: CustomValidationMetricsFn | None, reward_extra_infos: dict[str, list[Any]]
) -> dict[str, Any]:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.custom_validation_metrics_fn = custom_fn
    return trainer._val_metrics_update(
        np.array(DATA_SOURCES, dtype=object),
        SAMPLE_UIDS,
        reward_extra_infos,
        [np.array(SAMPLE_TURNS[:2]), np.array(SAMPLE_TURNS[2:])],
    )


def _v1_val_metrics(
    custom_fn: CustomValidationMetricsFn | None, reward_extra_infos: dict[str, list[Any]]
) -> dict[str, Any]:
    trainer = _StubV1Trainer.__new__(_StubV1Trainer)
    trainer.custom_validation_metrics_fn = custom_fn
    return trainer._val_metrics_update(list(DATA_SOURCES), SAMPLE_UIDS, reward_extra_infos, SAMPLE_TURNS)


both_trainers = pytest.mark.parametrize("val_metrics", [_legacy_val_metrics, _v1_val_metrics], ids=["legacy", "v1"])


@both_trainers
def test_custom_metrics_are_merged_with_builtin_metrics(metrics_module: str, val_metrics: ValMetricsUpdate) -> None:
    custom_fn = _load(metrics_module, "difficulty_metrics")
    reward_extra_infos = _reward_extra_infos()

    builtin = val_metrics(None, _reward_extra_infos())
    merged = val_metrics(custom_fn, reward_extra_infos)

    assert "val-core/math/acc/mean@2" in builtin
    assert "val-aux/num_turns/mean" in builtin
    assert {key: merged[key] for key in builtin} == builtin
    assert {key: merged[key] for key in merged.keys() - builtin.keys()} == {
        "val-aux/difficulty/easy/acc/mean": 1.0,
        "val-aux/difficulty/hard/acc/mean": pytest.approx(2 / 3),
    }

    last_call = custom_fn.__globals__["LAST_CALL"]
    assert type(last_call["data_sources"]) is list
    assert last_call["data_sources"] == DATA_SOURCES
    assert last_call["reward_extra_infos"] is reward_extra_infos


@both_trainers
def test_disabled_hook_only_produces_builtin_metrics(val_metrics: ValMetricsUpdate) -> None:
    assert _load(None, None) is None
    assert load_custom_validation_metrics_fn(OmegaConf.create({})) is None

    metrics = val_metrics(None, _reward_extra_infos())

    assert all(key.startswith(("val-core/", "val-aux/")) for key in metrics)
    assert not any("difficulty" in key for key in metrics)


@both_trainers
@pytest.mark.parametrize(
    ("fn_name", "error", "match"),
    [("not_a_dict", TypeError, "must return a dict"), ("builtin_collision", ValueError, "val-aux/num_turns/mean")],
)
def test_invalid_custom_metrics_fail(
    metrics_module: str, val_metrics: ValMetricsUpdate, fn_name: str, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        val_metrics(_load(metrics_module, fn_name), _reward_extra_infos())


@pytest.mark.parametrize(
    ("path", "name", "error"),
    [
        ("{module}", None, ValueError),
        ("{module}", "missing_fn", AttributeError),
        ("{module}", "NOT_CALLABLE", TypeError),
        ("/nonexistent/my_metrics.py", "difficulty_metrics", FileNotFoundError),
    ],
)
def test_bad_config_fails_at_load(metrics_module: str, path: str, name: str | None, error: type[Exception]) -> None:
    with pytest.raises(error):
        _load(path.format(module=metrics_module), name)


def _trainer_config(metrics_module: str, fn_name: str) -> DictConfig:
    overrides = [
        f"data.custom_validation_metrics.path={metrics_module}",
        f"data.custom_validation_metrics.name={fn_name}",
    ]
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def test_legacy_trainer_construction_fails_on_bad_name(metrics_module: str) -> None:
    config = _trainer_config(metrics_module, "missing_fn")
    with pytest.raises(AttributeError, match="missing_fn"):
        RayPPOTrainer(
            config=config,
            tokenizer=None,
            role_worker_mapping={Role.ActorRollout: object},
            resource_pool_manager=None,
        )


def test_v1_trainer_construction_fails_on_bad_name(metrics_module: str) -> None:
    config = _trainer_config(metrics_module, "missing_fn")
    with pytest.raises(AttributeError, match="missing_fn"):
        _StubV1Trainer(config)
