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


@pytest.mark.parametrize("mode", ["none", "tis"])
@pytest.mark.parametrize(
    "dtype, common_offset",
    [(torch.float32, 64.0), (torch.float32, 65536.0), (torch.bfloat16, 64.0)],
)
def test_low_entropy_support_gradient_matches_fp64_expected_score(mode, dtype, common_offset, device="cpu"):
    """Check the mathematical centered score when absolute log-normalizers lose precision."""
    from verl.trainer.ppo.score_centering import score_centering_shared_scores_from_logits

    # The finite support has three tokens. Larger logits outside it ensure this
    # checks the verified-support domain rather than a full-vocabulary score.
    initial = (
        torch.tensor([[0, 8, -11, 2, -12, -7, -4, 4, -5], [0, 6, -10, 1, -13, -8, -4, 5, -5]]) + common_offset
    ).to(device=device, dtype=dtype)
    ids = torch.tensor([[0, 2, 4, 6, 8]] * 2, device=device)
    labels = torch.tensor([0, 0], device=device)
    q_head = torch.tensor([[1 - 3e-5, 2e-5, 1e-5, 0, 0]] * 2, dtype=torch.float64, device=device).log().float()
    complete = torch.ones(2, dtype=torch.bool, device=device)
    advantages = torch.tensor([1.0, -0.75], dtype=torch.float64, device=device)
    rule = weight_fn(mode)

    # An independent expected-score objective: subtract E_q[w * log p^S]
    # with its weights frozen. Its gradient is the exact score-centering term;
    # no residual formula, vocabulary-gradient kernel or sparse scatter is used.
    ref_logits = initial.double().requires_grad_()
    finite = q_head.isfinite()
    head = ref_logits.gather(-1, ids).masked_fill(~finite, float("-inf"))
    shifted_head = head - head.max(-1, keepdim=True).values.detach()
    lp = (shifted_head - shifted_head.logsumexp(-1, keepdim=True)).masked_fill(~finite, 0.0)
    q64 = q_head.double().exp()
    torch.testing.assert_close(q64.sum(-1), q64.new_ones(2), rtol=0, atol=1e-10)
    assert torch.all((lp[:, 0].exp() > 0.9999) & (lp[:, 0].exp() < 1))
    with torch.no_grad():
        ratios = torch.where(finite, (lp - q_head.double()).exp(), torch.ones_like(lp))
        weighted_q = q64 * rule(ratios)
        sample_weights = rule((lp[:, 0] - q_head[:, 0].double()).exp())
    expected_loss = -(advantages * (sample_weights * lp[:, 0] - (weighted_q * lp).sum(-1))).mean()
    expected_grad = torch.autograd.grad(expected_loss, ref_logits)[0]

    actual_logits = initial.clone().requires_grad_()
    actual = score_centering_shared_scores_from_logits(actual_logits, ids, labels, q_head, complete, rule)
    actual_sample_weights = rule((actual["log_probs"].detach() - q_head[:, 0]).exp())
    actual_loss = -(advantages * (actual_sample_weights * actual["log_probs"] - actual["sc_correction"])).mean()
    actual_grad = torch.autograd.grad(actual_loss, actual_logits)[0].double()
    error = actual_grad - expected_grad
    # Tiny gradients need an explicit relative norm, alongside an absolute
    # bound that also permits one BF16 cast of the final score gradient.
    assert expected_grad.norm() > 0
    assert error.norm() / expected_grad.norm() < 0.01
    assert error.abs().max() < 8e-8
    outside_support = torch.ones_like(initial, dtype=torch.bool).scatter_(-1, ids, ~finite)
    assert torch.equal(actual_grad.masked_select(outside_support), torch.zeros_like(actual_grad[outside_support]))
    assert torch.isfinite(actual_grad).all()
    assert torch.equal(actual_logits.detach(), initial)
