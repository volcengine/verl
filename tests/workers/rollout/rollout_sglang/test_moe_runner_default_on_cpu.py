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
"""The MoE runner verl picks for SGLang when the user names none (sgl-project/sglang#27787)."""

import pytest

from verl.workers.rollout.sglang_rollout import utils


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
    assert utils.default_moe_runner_backend(None, {}) == expected


def test_quantized_rollouts_and_an_explicit_runner_keep_sglangs_choice(cuda_capability):
    cuda_capability((10, 0))
    assert utils.default_moe_runner_backend("fp8", {}) is None
    assert utils.default_moe_runner_backend(None, {"moe_runner_backend": "flashinfer_trtllm_routed"}) is None


def test_rocm_keeps_sglangs_choice(monkeypatch):
    monkeypatch.setattr(utils.torch.version, "hip", "6.4")
    monkeypatch.setattr(utils, "get_device_capability", lambda device_id=0: (10, 0))
    assert utils.default_moe_runner_backend(None, {}) is None
