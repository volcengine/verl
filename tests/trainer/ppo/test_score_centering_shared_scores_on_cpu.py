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
"""Independent oracle gates for the forthcoming FSDP shared SC scoring path."""

import pytest
import torch

from verl.trainer.ppo.score_centering import (
    score_centering_correction,
    score_centering_support_correction,
)


def weight_fn(mode):
    if mode == "none":
        return torch.ones_like
    if mode == "tis":
        return lambda ratio: ratio.clamp(max=2.0)
    return lambda ratio: torch.where((ratio >= 0.6) & (ratio <= 1.8), ratio, torch.zeros_like(ratio))


def fixture(dtype=torch.float32, device="cpu"):
    torch.manual_seed(8196)
    n, v, k = 6, 13, 5
    trainer_logits = (torch.randn(n, v, device=device) * 1.3).to(dtype)
    sampler_logits = torch.randn(n, v, device=device) * 1.8
    sizes = [2, 4, 8, 13, 5, 9]
    order = sampler_logits.argsort(-1, descending=True)
    keep = torch.arange(v, device=device).unsqueeze(0) < torch.tensor(sizes, device=device).unsqueeze(1)
    sampler_logits.scatter_(1, order, sampler_logits.gather(1, order).masked_fill(~keep, float("-inf")))
    log_q = torch.log_softmax(sampler_logits, -1)
    ids = order[:, :k]
    q_head = log_q.gather(-1, ids)
    complete = torch.tensor([True, True, False, False, True, False], device=device)
    # Rows 2/3/5 sample beyond the recorded head while staying in the true support.
    columns = torch.tensor([0, 3, 6, 10, 4, 7], device=device)
    labels = order.gather(-1, columns.unsqueeze(-1)).squeeze(-1)
    q_sampled = log_q.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    advantages = torch.tensor([0.8, -1.0, 1.2, 0.4, -0.6, 1.5], device=device)
    return trainer_logits, ids, labels, q_head, q_sampled, complete, advantages


def reference(logits, ids, labels, q_head, complete, rule):
    """Old full-V normalized head and sampled CE, then the per-row core route."""
    full_lp = torch.log_softmax(logits.float(), -1)
    head = full_lp.gather(-1, ids)
    fallback, sampler_mass, fallback_mass = score_centering_correction(head, q_head, rule)
    aligned, log_mass, _ = score_centering_support_correction(head, q_head, rule)
    correction = torch.where(complete, aligned, fallback)
    log_mass = torch.where(complete, log_mass, torch.zeros_like(log_mass))
    sampled = full_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - log_mass
    train_mass = torch.where(complete, log_mass.detach().exp(), fallback_mass)
    return {
        "sc_correction": correction,
        "log_probs": sampled,
        "sc_sampler_head_mass": sampler_mass,
        "sc_train_head_mass": train_mass,
    }


def direct_torch_oracle(logits, ids, labels, q_head, complete, rule):
    """Small-tensor oracle only; deliberately not a performance implementation."""
    full_lp = torch.log_softmax(logits.float(), -1)
    raw_head = logits.gather(-1, ids).float()
    finite = q_head.isfinite()
    support_lse = raw_head.masked_fill(~finite, float("-inf")).logsumexp(-1)
    support_lp = (raw_head - support_lse.unsqueeze(-1)).masked_fill(~finite, 0.0)
    head_full_lp = full_lp.gather(-1, ids)
    selected_lp = torch.where(complete.unsqueeze(-1), support_lp, head_full_lp)
    with torch.no_grad():
        q = q_head.exp()
        selected_p = selected_lp.exp() * (~complete.unsqueeze(-1) | finite)
        ratio = torch.where(finite, (selected_lp - q_head).exp(), torch.ones_like(q))
        weighted = q * rule(ratio)
        p_mass, q_mass = selected_p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(1e-6) / (1 - p_mass).clamp_min(1e-6)
        alpha = rho * rule(1 / rho)
        fallback_residual = weighted - alpha.unsqueeze(-1) * selected_p
        support_residual = weighted - weighted.sum(-1, keepdim=True) * selected_p
        residual = torch.where(complete.unsqueeze(-1), support_residual, fallback_residual)
    correction = (residual * selected_lp).sum(-1)
    full_sampled = full_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    support_sampled = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float() - support_lse
    sampled = torch.where(complete, support_sampled, full_sampled)
    with torch.no_grad():
        head_p = head_full_lp.exp()
        train_mass = torch.where(complete, (head_p * finite).sum(-1), head_p.sum(-1))
    return {
        "sc_correction": correction,
        "log_probs": sampled,
        "sc_sampler_head_mass": q_head.exp().sum(-1),
        "sc_train_head_mass": train_mass,
    }


def loss(outputs, logits, q_sampled, advantages, rule, auxiliary):
    lp = outputs["log_probs"]
    weights = rule((lp.detach() - q_sampled).exp())
    result = -(advantages * (weights * lp - outputs["sc_correction"])).mean()
    if auxiliary:
        full_lp = torch.log_softmax(logits.float(), -1)
        entropy = -(full_lp.exp() * full_lp).sum(-1).mean()
        # Preserve current KL consumer semantics; reference LP remains a fixed scalar.
        diff = -2.0 - lp
        kl = (diff.exp() - 1 - diff).mean()
        result = result - 0.02 * entropy + 0.01 * kl
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_direct_support_score_oracle_has_loss_and_gradient_parity(dtype, mode, auxiliary):
    initial, ids, labels, q_head, q_sampled, complete, advantages = fixture(dtype)
    rule = weight_fn(mode)
    ref_logits = initial.clone().requires_grad_(True)
    ref = reference(ref_logits, ids, labels, q_head, complete, rule)
    expected_loss = loss(ref, ref_logits, q_sampled, advantages, rule, auxiliary)
    expected_loss.backward()
    new_logits = initial.clone().requires_grad_(True)
    got = direct_torch_oracle(new_logits, ids, labels, q_head, complete, rule)
    actual_loss = loss(got, new_logits, q_sampled, advantages, rule, auxiliary)
    actual_loss.backward()
    for name in got:
        torch.testing.assert_close(got[name], ref[name], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(
        new_logits.grad,
        ref_logits.grad,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        atol=2e-3 if dtype == torch.bfloat16 else 2e-6,
    )
    assert torch.isfinite(new_logits.grad).all()
    assert torch.equal(new_logits.detach(), initial)


@pytest.mark.parametrize("mode", ["none", "tis", "icepop"])
@pytest.mark.parametrize("noncontiguous_receipt", [False, True])
def test_shared_helper_candidate_against_independent_oracle(mode, noncontiguous_receipt):
    # The implementation will be added after rebasing onto the automatic feature.
    from verl.trainer.ppo.score_centering import score_centering_shared_scores_from_logits

    initial, ids, labels, q_head, q_sampled, complete, advantages = fixture()
    if noncontiguous_receipt:
        backing = torch.zeros(complete.numel() * 2, dtype=torch.bool)
        backing[::2] = complete
        complete = backing[::2]
        assert not complete.is_contiguous()
    rule = weight_fn(mode)
    reference_logits = initial.clone().requires_grad_(True)
    expected = reference(reference_logits, ids, labels, q_head, complete, rule)
    expected_loss = loss(expected, reference_logits, q_sampled, advantages, rule, True)
    expected_loss.backward()
    candidate_logits = initial.clone().requires_grad_(True)
    actual = score_centering_shared_scores_from_logits(
        candidate_logits, ids, labels, q_head, complete, rule, chunk_size=2
    )
    actual_loss = loss(actual, candidate_logits, q_sampled, advantages, rule, True)
    actual_loss.backward()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(candidate_logits.grad, reference_logits.grad, rtol=1e-5, atol=2e-6)
    assert torch.equal(candidate_logits.detach(), initial)
