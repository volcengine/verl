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
"""CUDA/Triton regression gates for mixed shared score-centering scores."""

import pytest
import torch

from tests.trainer.ppo.test_score_centering_shared_scores_on_cpu import fixture, loss, reference, weight_fn
from tests.trainer.ppo.test_score_centering_shared_scores_on_cpu import (
    test_low_entropy_support_gradient_matches_fp64_expected_score as check_low_entropy_expected_score,
)
from verl.trainer.ppo.score_centering import score_centering_shared_scores_from_logits

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scorer regression requires a GPU.")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
@pytest.mark.parametrize("strided_inputs", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_shared_cuda_scores_match_mixed_route_oracle(dtype, mode, strided_inputs, auxiliary):
    initial, ids, labels, q_head, q_sampled, complete, advantages = fixture(dtype, "cuda")
    rule = weight_fn(mode)
    ref_logits = initial.clone().requires_grad_(True)
    expected = reference(ref_logits, ids, labels, q_head, complete, rule)
    expected_loss = loss(expected, ref_logits, q_sampled, advantages, rule, auxiliary)
    expected_loss.backward()
    if strided_inputs:
        storage = torch.zeros(initial.shape[0], initial.shape[1] * 2, device="cuda", dtype=dtype)
        storage[:, ::2] = initial
        actual_logits = storage[:, ::2].requires_grad_(True)
        receipts = torch.zeros(complete.numel() * 2, device="cuda", dtype=torch.bool)
        receipts[::2] = complete
        complete = receipts[::2]
        assert actual_logits.stride(-1) == 2 and complete.stride(0) == 2
    else:
        actual_logits = initial.clone().requires_grad_(True)
    actual = score_centering_shared_scores_from_logits(actual_logits, ids, labels, q_head, complete, rule, chunk_size=2)
    actual_loss = loss(actual, actual_logits, q_sampled, advantages, rule, auxiliary)
    actual_loss.backward()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(
        actual_logits.grad,
        ref_logits.grad,
        rtol=0.03 if dtype == torch.bfloat16 else 3e-5,
        atol=2e-3 if dtype == torch.bfloat16 else 3e-6,
    )
    assert torch.isfinite(actual_logits.grad).all()
    assert torch.equal(actual_logits.detach(), initial)
    if not auxiliary:
        support = torch.zeros_like(initial, dtype=torch.bool).scatter_(-1, ids, q_head.isfinite())
        outside = actual_logits.grad.masked_fill(support | ~complete.unsqueeze(-1), 0)
        assert torch.equal(outside, torch.zeros_like(outside))


@pytest.mark.parametrize("mode", ["none", "tis"])
@pytest.mark.parametrize(
    "dtype, common_offset",
    [(torch.float32, 64.0), (torch.float32, 65536.0), (torch.bfloat16, 64.0)],
)
def test_shared_cuda_low_entropy_gradient_matches_fp64_expected_score(mode, dtype, common_offset):
    check_low_entropy_expected_score(mode, dtype, common_offset, device="cuda")
