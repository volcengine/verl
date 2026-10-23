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
"""configure_bypass_mode selects the actor loss in the config the workers are created from;
the bypass loss handles the micro-batches it then receives."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.rollout_corr_helper import configure_bypass_mode


def _config(bypass_mode: bool, loss_mode: str, actor_rc=None, megatron=None, **rollout_correction):
    policy_loss = {"loss_mode": loss_mode}
    if actor_rc is not None:
        policy_loss["rollout_correction"] = actor_rc
    config = OmegaConf.create(
        {
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": bypass_mode,
                    "loss_type": "ppo_clip",
                    "rollout_is": None,
                    "rollout_is_threshold": 2.0,
                    **rollout_correction,
                }
            },
            "actor_rollout_ref": {"actor": {"policy_loss": policy_loss, "megatron": megatron or {}}},
        }
    )
    OmegaConf.set_struct(config, True)  # as composed by Hydra
    return config


@pytest.mark.parametrize(
    "bypass_mode,loss_mode,expected",
    [
        (True, "vanilla", "bypass_mode"),
        (True, "bypass_mode", "bypass_mode"),
        (True, "gspo", "gspo"),
        (False, "vanilla", "vanilla"),
        (False, "gspo", "gspo"),
    ],
    ids=["bypass-from-default", "bypass-already-set", "bypass-explicit-loss", "decoupled", "decoupled-other-loss"],
)
def test_actor_loss(bypass_mode, loss_mode, expected):
    config = _config(bypass_mode, loss_mode)
    configure_bypass_mode(config)
    policy_loss = config.actor_rollout_ref.actor.policy_loss
    assert policy_loss.loss_mode == expected
    if expected == "bypass_mode":
        assert policy_loss.rollout_correction == config.algorithm.rollout_correction
    else:
        assert "rollout_correction" not in policy_loss


@pytest.mark.parametrize(
    "loss_mode,actor_rc,rollout_correction,match",
    [
        ("gspo", None, {"rollout_is": "token"}, "rollout_is has no effect"),
        ("gspo", None, {"loss_type": "reinforce"}, "loss_type has no effect"),
        ("bypass_mode", {"loss_type": "reinforce"}, {}, "rollout_correction.loss_type differs"),
        (
            "bypass_mode",
            {"rollout_is_threshold": "0.5_2.0"},
            {"rollout_is": "token"},
            "rollout_correction.rollout_is_threshold differs",
        ),
        (
            "bypass_mode",
            {"rollout_rs_threshold": "0.5_3.0"},
            {"rollout_rs": "token_k1", "rollout_rs_threshold": 2.0},
            "rollout_correction.rollout_rs_threshold differs",
        ),
        (
            "vanilla",
            None,
            {
                "rollout_is": "token",
                "rollout_is_batch_normalize": True,
                "megatron": {"pipeline_model_parallel_size": 2},
            },
            "not supported with Megatron pipeline",
        ),
        (
            "vanilla",
            None,
            {
                "rollout_is": "sequence",
                "rollout_is_batch_normalize": True,
                "megatron": {"dynamic_context_parallel": True},
            },
            "dynamic context parallelism",
        ),
    ],
    ids=[
        "explicit-loss-with-is",
        "explicit-loss-with-reinforce",
        "conflicting-actor-mirror",
        "different-is-bounds",
        "different-rs-bounds",
        "batch-normalize-with-pipeline-parallelism",
        "batch-normalize-with-dynamic-cp",
    ],
)
def test_rejected(loss_mode, actor_rc, rollout_correction, match):
    with pytest.raises(ValueError, match=match):
        configure_bypass_mode(_config(True, loss_mode, actor_rc, **rollout_correction))


@pytest.mark.parametrize(
    "actor_rc,rollout_correction",
    [
        ({"loss_type": "ppo_clip", "rollout_is": "token"}, {"rollout_is": "token"}),
        ("${algorithm.rollout_correction}", {"rollout_is": "token"}),
        ({"rollout_is_threshold": "2.0"}, {"rollout_is": "token"}),
        ({"rollout_is_threshold": "0.50_2"}, {"rollout_is": "token", "rollout_is_threshold": "0.5_2.0"}),
        ({"rollout_rs_threshold": "0.5_2.0"}, {"rollout_rs": "token_k1", "rollout_rs_threshold": 2.0}),
        ({"rollout_rs_threshold": "2.0"}, {"rollout_rs": None, "rollout_rs_threshold": 2.0}),
    ],
    ids=[
        "hand-written",
        "interpolated",
        "is-threshold-as-string",
        "is-range-spelling",
        "rs-k1-range-of-scalar",
        "inert-rs-threshold",
    ],
)
def test_matching_actor_mirror_is_accepted(actor_rc, rollout_correction):
    config = _config(True, "bypass_mode", actor_rc, **rollout_correction)
    configure_bypass_mode(config)
    assert config.actor_rollout_ref.actor.policy_loss.rollout_correction == config.algorithm.rollout_correction


@pytest.mark.parametrize(
    "loss_type,correction",
    [
        ("ppo_clip", {"rollout_is": "token"}),
        ("reinforce", {"rollout_is": "token"}),
        ("reinforce", {"rollout_is": "token", "rollout_is_threshold": "1.5_2.0"}),
        ("reinforce", {"rollout_is": "sequence", "rollout_is_batch_normalize": True}),
        ("reinforce", {"rollout_is": "token", "rollout_rs": "token_k1,seq_mean_k3", "rollout_rs_threshold": 2.0}),
    ],
    ids=["ppo-clip", "reinforce", "icepop-range", "seq-batch-normalize", "rejection"],
)
def test_padding_only_micro_batch_has_zero_loss(loss_type, correction):
    # Failed rollout groups are padded with masked samples; a micro-batch may hold only those.
    import math

    import torch

    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.trainer.ppo.core_algos import compute_policy_loss_bypass_mode
    from verl.workers.config import ActorConfig, PolicyLossConfig

    rollout_correction = RolloutCorrectionConfig(bypass_mode=True, loss_type=loss_type, **correction)
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rollout_correction),
        loss_agg_mode="token-mean",
    )
    config.global_batch_info.update(dp_size=1, batch_num_tokens=8, global_batch_size=2, loss_scale_factor=None)
    log_prob = torch.zeros(1, 4, requires_grad=True)
    advantages = torch.ones(1, 4)
    loss, metrics = compute_policy_loss_bypass_mode(
        log_prob.detach() - 0.1, log_prob, advantages, torch.zeros(1, 4, dtype=torch.bool), "token-mean", config
    )
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(log_prob.grad) == 0
    # Same metrics as a real micro-batch, so DP ranks aggregate equal-length lists.
    _, real_metrics = compute_policy_loss_bypass_mode(
        log_prob.detach() - 0.1, log_prob, advantages, torch.ones(1, 4, dtype=torch.bool), "token-mean", config
    )
    assert metrics.keys() == real_metrics.keys()
    assert all(math.isfinite(float(value)) for value in metrics.values())
