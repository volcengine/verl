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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import VeOmniEngineConfig


def test_low_precision_communication_is_opt_in():
    assert VeOmniEngineConfig().low_precision_reduce_scatter_comm is False


@pytest.mark.parametrize("enabled", [None, 0, 1, "false", "true"])
def test_low_precision_communication_rejects_non_boolean_values(enabled):
    with pytest.raises(ValueError, match="low_precision_reduce_scatter_comm.*bool"):
        VeOmniEngineConfig(low_precision_reduce_scatter_comm=enabled)


@pytest.mark.parametrize("enabled", [False, True])
def test_actor_communication_setting_does_not_enable_reference_communication(enabled):
    config_dir = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=veomni",
                f"actor_rollout_ref.actor.veomni.low_precision_reduce_scatter_comm={str(enabled).lower()}",
            ],
        )

    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor.veomni)
    reference = omega_conf_to_dataclass(config.actor_rollout_ref.ref.veomni)

    assert isinstance(actor, VeOmniEngineConfig)
    assert actor.low_precision_reduce_scatter_comm is enabled
    assert actor.mixed_precision is True
    assert actor.forward_only is False
    assert isinstance(reference, VeOmniEngineConfig)
    assert reference.low_precision_reduce_scatter_comm is False
    assert reference.forward_only is True
