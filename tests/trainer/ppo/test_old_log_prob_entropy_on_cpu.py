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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.v1 import trainer_base
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig


def _config(enabled: bool | None, entropy_coeff: float = 0.0) -> DictConfig:
    actor = {
        "calculate_entropy": False,
        "entropy_coeff": entropy_coeff,
        "loss_agg_mode": "token-mean",
        "loss_scale_factor": None,
        "calculate_sum_pi_squared": True,
    }
    if enabled is not None:
        actor["calculate_old_log_prob_entropy"] = enabled
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": actor, "rollout": {"temperature": 1.0, "calculate_log_probs": True}},
            "algorithm": {},
        }
    )


def _nested(rows: list[list[float]]) -> torch.Tensor:
    return torch.nested.as_nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("entropy_coeff", [0.0, 0.01])
def test_ray_old_log_prob_optional_entropy(enabled: bool | None, entropy_coeff: float) -> None:
    requested: list[bool] = []
    routes = torch.tensor([[[1]], [[2]]])

    def compute_log_prob(data: TensorDict) -> TensorDict:
        calculate_entropy = tu.get_non_tensor_data(data, "calculate_entropy", None)
        requested.append(calculate_entropy)
        # Model outputs are shifted once by the real response-extraction helper.
        result = {
            "log_probs": _nested([[10.0, 11.0, 12.0, 13.0, 14.0], [20.0, 21.0, 22.0]]),
            "sum_pi_squared": _nested([[0.1] * 5, [0.2] * 3]),
            "routed_experts": routes,
        }
        if calculate_entropy:
            result["entropy"] = _nested([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        output = tu.get_tensordict(result)
        tu.assign_non_tensor(output, metrics={"mfu": 0.25})
        return output

    batch = DataProto.from_dict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5], [0, 6, 7, 8, 0]]),
            "prompts": torch.tensor([[1, 2], [0, 6]]),
            "responses": torch.tensor([[3, 4, 5], [7, 8, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 0, 1], [1, 1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 0]]),
        }
    )
    trainer = SimpleNamespace(
        config=_config(enabled, entropy_coeff), actor_rollout_wg=SimpleNamespace(compute_log_prob=compute_log_prob)
    )
    output, mfu = RayPPOTrainer._compute_old_log_prob(trainer, batch)
    assert requested == [enabled is not False]
    torch.testing.assert_close(output.batch["old_log_probs"], torch.tensor([[11.0, 12.0, 13.0], [20.0, 21.0, 0.0]]))
    torch.testing.assert_close(output.batch["routed_experts"], routes)
    torch.testing.assert_close(output.batch["sum_pi_squared"], torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.0]]))
    assert mfu == 0.25
    assert ("entropys" in output.batch) is (enabled is not False)
    if enabled is not False:
        torch.testing.assert_close(output.batch["entropys"], torch.tensor([[2.0, 3.0, 4.0], [6.0, 7.0, 0.0]]))


@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("entropy_coeff", [0.0, 0.01])
def test_v1_old_log_prob_optional_entropy(
    enabled: bool | None, entropy_coeff: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = trainer_base.KVBatchMeta(partition_id="train", keys=["a", "b"], tags=[{}, {}])
    stored = TensorDict(
        {
            "response_mask": _nested([[1.0, 0.0, 1.0], [1.0, 1.0]]),
            "responses": _nested([[3.0, 4.0, 5.0], [7.0, 8.0]]),
            "rollout_log_probs": _nested([[-0.1, -0.2, -0.3], [-0.4, -0.5]]),
        },
        batch_size=[2],
    )
    fetched: list[str] = []
    written: list[str] = []

    def compute_log_prob(meta: trainer_base.KVBatchMeta) -> trainer_base.KVBatchMeta:
        stored["log_probs"] = _nested([[0.0, -0.1, -0.2, -0.3, 0.0], [-0.4, -0.5, 0.0]])
        if meta.extra_info["calculate_entropy"]:
            stored["entropy"] = _nested([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        return meta

    def get(*, keys: list[str], partition_id: str, select_fields: list[str]) -> TensorDict:
        fetched.extend(select_fields)
        return stored.select(*select_fields).clone()

    def put(*, keys: list[str], partition_id: str, fields: TensorDict) -> trainer_base.KVBatchMeta:
        written.extend(fields.keys())
        stored.update(fields)
        return batch

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    trainer = SimpleNamespace(
        config=_config(enabled, entropy_coeff), actor_rollout_wg=SimpleNamespace(compute_log_prob=compute_log_prob)
    )
    metrics: dict = {}
    result = trainer_base.PPOTrainer._compute_old_log_prob(trainer, batch, metrics)
    assert result is batch
    assert batch.extra_info["calculate_entropy"] is (enabled is not False)
    assert ("entropy" in fetched) is (enabled is not False)
    assert ("entropy" in written) is (enabled is not False)
    assert ("actor/entropy" in metrics) is (enabled is not False)
    if enabled is not False:
        assert metrics["actor/entropy"] == pytest.approx((2 + 4 + 6 + 7) / 4)
    assert metrics["training/rollout_probs_diff_max"] == pytest.approx(0.0)
    torch.testing.assert_close(stored["old_log_probs"].values(), torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5]))


@pytest.mark.parametrize("enabled", [True, False])
def test_v1_bypass_skips_old_log_prob_forward(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = trainer_base.KVBatchMeta(partition_id="train", keys=["a"], tags=[{}])
    rollout = _nested([[-0.2, -0.3]])
    get = Mock(return_value=TensorDict({"rollout_log_probs": rollout}, batch_size=[1]))
    put = Mock()
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)
    config = _config(enabled)
    config.algorithm.rollout_correction = {"bypass_mode": True}
    worker = Mock()
    trainer = SimpleNamespace(config=config, actor_rollout_wg=worker)
    metrics: dict = {}
    assert trainer_base.PPOTrainer._compute_old_log_prob(trainer, batch, metrics) is batch
    worker.compute_log_prob.assert_not_called()
    torch.testing.assert_close(put.call_args.kwargs["fields"]["old_log_probs"].values(), rollout.values())
    assert metrics == {}


def test_actor_config_preserves_default_old_log_prob_entropy() -> None:
    assert ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True).calculate_old_log_prob_entropy is True
