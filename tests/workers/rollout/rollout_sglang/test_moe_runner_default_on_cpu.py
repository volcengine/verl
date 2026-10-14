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
"""The MoE runner verl picks for SGLang when the user leaves it to auto (sgl-project/sglang#27787)."""

import types

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.sglang_rollout import utils

BF16_CHECKPOINT = types.SimpleNamespace()  # hf_config without a quantization_config
FP8_CHECKPOINT = types.SimpleNamespace(quantization_config={"quant_method": "fp8"})


@pytest.fixture
def cuda_capability(monkeypatch):
    monkeypatch.setattr(utils.torch.version, "hip", None)

    def set_capability(capability):
        monkeypatch.setattr(utils, "get_device_capability", lambda device_id=0: capability)

    return set_capability


@pytest.mark.parametrize(
    "capability, expected",
    [((10, 0), "triton"), ((10, 3), "triton"), ((9, 0), None), ((12, 0), None), ((None, None), None)],
)
def test_bf16_rollouts_use_the_triton_runner_on_blackwell_only(cuda_capability, capability, expected):
    cuda_capability(capability)
    assert utils.default_moe_runner_backend(None, {}, BF16_CHECKPOINT) == expected


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {},
        {"moe_runner_backend": "auto"},
        {"moe_runner_backend": None},
        {"moe_a2a_backend": "none"},
        OmegaConf.create({"moe_runner_backend": "auto"}),  # engine_kwargs comes from the Hydra config
    ],
)
def test_a_runner_left_to_auto_without_an_a2a_backend_gets_triton(cuda_capability, engine_kwargs):
    cuda_capability((10, 0))
    assert utils.default_moe_runner_backend(None, engine_kwargs, BF16_CHECKPOINT) == "triton"


@pytest.mark.parametrize(
    "quantization, engine_kwargs, hf_config",
    [
        ("fp8", {}, BF16_CHECKPOINT),  # quantized rollout
        (None, {}, FP8_CHECKPOINT),  # quantized checkpoint, which SGLang detects on its own
        (None, {"moe_runner_backend": "flashinfer_trtllm_routed"}, BF16_CHECKPOINT),  # explicit runner
        (None, {"moe_a2a_backend": "deepep"}, BF16_CHECKPOINT),  # SGLang does not pick flashinfer_trtllm then
        (None, OmegaConf.create({"moe_a2a_backend": "deepep"}), BF16_CHECKPOINT),
    ],
)
def test_sglangs_choice_is_kept_outside_the_failing_case(cuda_capability, quantization, engine_kwargs, hf_config):
    cuda_capability((10, 0))
    assert utils.default_moe_runner_backend(quantization, engine_kwargs, hf_config) is None


def test_rocm_keeps_sglangs_choice(monkeypatch):
    monkeypatch.setattr(utils.torch.version, "hip", "6.4")
    monkeypatch.setattr(utils, "get_device_capability", lambda device_id=0: (10, 0))
    assert utils.default_moe_runner_backend(None, {}, BF16_CHECKPOINT) is None
