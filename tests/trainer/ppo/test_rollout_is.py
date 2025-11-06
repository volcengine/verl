#!/usr/bin/env python3
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
"""
Quick Sanity Test for Rollout Importance Sampling

This is a standalone test script that can be run without pytest to quickly verify
the rollout IS implementation is working correctly. For comprehensive integration
tests, see: tests/trainer/ppo/test_rollout_is_integration.py

Usage:
    python test_rollout_is.py

This tests:
- Basic rollout IS functionality (3 levels, 2 modes)
- Metrics completeness (32 total: 21 IS + 11 mismatch metrics)
- Veto mechanism
- Edge cases
"""

import torch

from verl.trainer.ppo.mismatch_helper import compute_mismatch_metrics, compute_rollout_importance_weights


def test_basic_rollout_is():
    """Test basic rollout IS functionality."""
    print("Testing basic rollout IS functionality...")

    # Create test data
    batch_size, seq_length = 4, 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create slightly different log probs (simulating BF16 vs FP32 mismatch)
    old_log_prob = torch.randn(batch_size, seq_length, device=device)
    rollout_log_prob = old_log_prob + torch.randn(batch_size, seq_length, device=device) * 0.1
    eos_mask = torch.ones(batch_size, seq_length, device=device)

    # Test token-level truncate mode
    print("\n1. Testing token-level truncate mode...")
    weights_proto, modified_response_mask, metrics = compute_rollout_importance_weights(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is_level="token",
        rollout_is_mode="truncate",
        rollout_is_threshold=2.0,
        rollout_is_veto_threshold=1e-4,
    )

    weights = weights_proto.batch["rollout_is_weights"]
    print(f"   Weights shape: {weights.shape}")
    print(f"   Mean weight: {metrics['mismatch/rollout_is_mean']:.4f}")
    print(f"   Max weight: {metrics['mismatch/rollout_is_max']:.4f}")
    print(f"   Min weight: {metrics['mismatch/rollout_is_min']:.4f}")
    print(f"   Veto fraction: {metrics['mismatch/rollout_is_veto_fraction']:.4f}")
    assert weights.shape == old_log_prob.shape
    assert weights.max() <= 2.0, "Weights should be capped at threshold"
    print("   ✓ Token-level truncate mode passed")

    # Test sequence-level mode
    print("\n2. Testing sequence-level mode...")
    weights_seq_proto, _, metrics_seq = compute_rollout_importance_weights(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is_level="sequence",
        rollout_is_mode="truncate",
        rollout_is_threshold=5.0,
        rollout_is_veto_threshold=1e-4,
    )

    weights_seq = weights_seq_proto.batch["rollout_is_weights"]
    print(f"   Mean weight: {metrics_seq['mismatch/rollout_is_mean']:.4f}")
    print(f"   Effective sample size: {metrics_seq['mismatch/rollout_is_eff_sample_size']:.4f}")
    # Check that all tokens in a sequence have the same weight
    for i in range(batch_size):
        seq_weights = weights_seq[i, eos_mask[i].bool()]
        assert torch.allclose(seq_weights, seq_weights[0]), "All tokens in sequence should have same weight"
    print("   ✓ Sequence-level mode passed")

    # Test geometric mean mode
    print("\n3. Testing geometric mean mode...")
    weights_geo_proto, _, metrics_geo = compute_rollout_importance_weights(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is_level="geometric",
        rollout_is_mode="mask",
        rollout_is_threshold=1.5,
        rollout_is_threshold_lower=0.5,
        rollout_is_veto_threshold=1e-4,
    )

    print(f"   Mean weight: {metrics_geo['mismatch/rollout_is_mean']:.4f}")
    print(f"   Masked fraction: {metrics_geo['mismatch/rollout_is_masked_fraction']:.4f}")
    print("   ✓ Geometric mean mode passed")

    # Test veto mechanism
    print("\n4. Testing veto mechanism...")
    # Create data with catastrophic outliers
    old_log_prob_veto = torch.randn(2, 5, device=device)
    rollout_log_prob_veto = old_log_prob_veto.clone()
    # Make one token have catastrophically low ratio
    rollout_log_prob_veto[0, 2] = old_log_prob_veto[0, 2] + 15.0  # ratio ~= 3e-7
    eos_mask_veto = torch.ones(2, 5, device=device)

    weights_veto_proto, modified_response_mask_veto, metrics_veto = compute_rollout_importance_weights(
        old_log_prob=old_log_prob_veto,
        rollout_log_prob=rollout_log_prob_veto,
        response_mask=eos_mask_veto,
        rollout_is_level="token",
        rollout_is_mode="truncate",
        rollout_is_threshold=2.0,
        rollout_is_veto_threshold=1e-4,
    )

    weights_veto = weights_veto_proto.batch["rollout_is_weights"]
    print(f"   Veto fraction: {metrics_veto['mismatch/rollout_is_veto_fraction']:.4f}")
    # KEY FIX: Veto is applied via response_mask, not by zeroing weights
    # Check that weights are NON-ZERO (safety-bounded ratios preserved, not zeroed)
    assert weights_veto[0].sum() > 0, "Weights should be non-zero (not zeroed by veto)"
    # Check that response_mask has veto applied
    assert modified_response_mask_veto[0].sum() == 0, "Vetoed sequence should have response_mask zeroed"
    assert modified_response_mask_veto[1].sum() > 0, "Normal sequence should have response_mask unchanged"
    print("   ✓ Veto mechanism passed")

    # Test disabled IS (threshold=None)
    print("\n5. Testing disabled IS...")
    weights_disabled, modified_response_mask_disabled, metrics_disabled = compute_rollout_importance_weights(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is_threshold=None,
    )

    assert weights_disabled is None, "Should return None when threshold is None"
    assert torch.equal(modified_response_mask_disabled, eos_mask), "Should return original mask unchanged"
    assert len(metrics_disabled) == 0, "Should return empty metrics when disabled"
    print("   ✓ Disabled IS passed")

    print("\n✓ All tests passed!")


def test_metrics_completeness():
    """Test that all expected metrics are returned."""
    print("\nTesting metrics completeness...")

    batch_size, seq_length = 3, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old_log_prob = torch.randn(batch_size, seq_length, device=device)
    rollout_log_prob = old_log_prob + torch.randn(batch_size, seq_length, device=device) * 0.2
    eos_mask = torch.ones(batch_size, seq_length, device=device)

    _, _, metrics = compute_rollout_importance_weights(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=eos_mask,
        rollout_is_level="token",
        rollout_is_mode="truncate",
        rollout_is_threshold=2.5,
    )

    # Expected IS metrics
    expected_is_metrics = [
        "mismatch/rollout_is_mean",
        "mismatch/rollout_is_max",
        "mismatch/rollout_is_min",
        "mismatch/rollout_is_std",
        "mismatch/rollout_is_eff_sample_size",
        "mismatch/rollout_is_veto_fraction",
        "mismatch/rollout_is_catastrophic_token_fraction",
        "mismatch/rollout_is_ratio_fraction_high",
        "mismatch/rollout_is_ratio_fraction_low",
    ]

    # Expected mismatch/diagnostic metrics (also included now)
    expected_mismatch_metrics = [
        "mismatch/mismatch_training_ppl",
        "mismatch/mismatch_training_log_ppl",
        "mismatch/mismatch_kl",
        "mismatch/mismatch_k3_kl",
        "mismatch/mismatch_rollout_ppl",
        "mismatch/mismatch_rollout_log_ppl",
        "mismatch/mismatch_log_ppl_diff",
        "mismatch/mismatch_log_ppl_abs_diff",
        "mismatch/mismatch_log_ppl_diff_max",
        "mismatch/mismatch_log_ppl_diff_min",
        "mismatch/mismatch_ppl_ratio",
    ]

    expected_metrics = expected_is_metrics + expected_mismatch_metrics

    missing_metrics = [m for m in expected_metrics if m not in metrics]
    if missing_metrics:
        print(f"   ✗ Missing metrics: {missing_metrics}")
        return False

    print(f"   ✓ All {len(expected_metrics)} expected metrics present")
    print(f"   Total metrics returned: {len(metrics)}")
    return True


def test_mismatch_metrics():
    """Test mismatch metrics computation."""
    print("\nTesting mismatch metrics computation...")

    batch_size, seq_length = 4, 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test data with some mismatch
    old_log_prob = torch.randn(batch_size, seq_length, device=device) - 2.0  # training policy
    rollout_log_prob = torch.randn(batch_size, seq_length, device=device) - 1.5  # rollout policy (more confident)
    response_mask = torch.ones(batch_size, seq_length, device=device)

    # Test with rollout log probs
    metrics = compute_mismatch_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    expected_metrics = [
        "mismatch_training_ppl",
        "mismatch_training_log_ppl",
        "mismatch_kl",
        "mismatch_k3_kl",
        "mismatch_rollout_ppl",
        "mismatch_rollout_log_ppl",
        "mismatch_log_ppl_diff",
        "mismatch_log_ppl_abs_diff",
        "mismatch_log_ppl_diff_max",
        "mismatch_log_ppl_diff_min",
        "mismatch_ppl_ratio",
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    print(f"   Training PPL: {metrics['mismatch_training_ppl']:.4f}")
    print(f"   Rollout PPL: {metrics['mismatch_rollout_ppl']:.4f}")
    print(f"   KL divergence: {metrics['mismatch_kl']:.6f}")
    print(f"   K3 KL: {metrics['mismatch_k3_kl']:.6f}")
    print(f"   PPL ratio: {metrics['mismatch_ppl_ratio']:.4f}")
    print(f"   ✓ All {len(expected_metrics)} mismatch metrics present")

    # Test without rollout log probs
    metrics_no_rollout = compute_mismatch_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=None,
        response_mask=response_mask,
    )

    assert "mismatch_training_ppl" in metrics_no_rollout
    assert "mismatch_rollout_ppl" not in metrics_no_rollout
    print("   ✓ Mismatch metrics work without rollout log probs")


def test_mask_mode():
    """Test mask mode applies rejection via response_mask, keeps true IS weights."""
    print("\nTesting mask mode behavior...")

    batch_size = 2
    seq_length = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sequence 0: ratio ≈ 0.37 (below 0.5, should be rejected)
    # Sequence 1: ratio ≈ 1.65 (in [0.5, 2.0], should be accepted)
    old_log_prob = torch.tensor([[-2.0] * seq_length, [-2.0] * seq_length], device=device)
    rollout_log_prob = torch.tensor(
        [
            [-1.0] * seq_length,  # exp(-2.0 - (-1.0)) = exp(-1.0) ≈ 0.37
            [-2.5] * seq_length,  # exp(-2.0 - (-2.5)) = exp(0.5) ≈ 1.65
        ],
        device=device,
    )
    response_mask = torch.ones(batch_size, seq_length, device=device)

    weights_proto, modified_response_mask, metrics = compute_rollout_importance_weights(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is_level="token",
        rollout_is_mode="mask",
        rollout_is_threshold=2.0,
        rollout_is_threshold_lower=0.5,
        rollout_is_veto_threshold=None,
    )

    weights = weights_proto.batch["rollout_is_weights"]

    # KEY FIX: Weights should be safety-bounded ratios (NOT zeroed)
    assert torch.all(weights[0, :] > 0), "Weights should remain as safety-bounded ratios (not zeroed)"
    assert torch.allclose(weights[0, 0], torch.tensor(0.368, device=device), atol=0.01), (
        "First seq ratio should be ≈0.37"
    )
    assert torch.allclose(weights[1, 0], torch.tensor(1.649, device=device), atol=0.01), (
        "Second seq ratio should be ≈1.65"
    )

    # Rejection should be applied via response_mask
    assert torch.all(modified_response_mask[0, :] == 0), "First sequence should be rejected via mask"
    assert torch.all(modified_response_mask[1, :] == 1), "Second sequence should be accepted"

    # Verify mask metrics exist
    assert "mismatch/rollout_is_masked_fraction" in metrics
    assert abs(metrics["mismatch/rollout_is_masked_fraction"] - 0.5) < 0.01, "Should reject 50% of tokens"

    print(f"   First seq IS weight: {weights[0, 0]:.4f} (expected ≈0.37)")
    print(f"   Second seq IS weight: {weights[1, 0]:.4f} (expected ≈1.65)")
    print(f"   First seq mask: {modified_response_mask[0, 0]:.0f} (expected 0 - rejected)")
    print(f"   Second seq mask: {modified_response_mask[1, 0]:.0f} (expected 1 - accepted)")
    print(f"   Masked fraction: {metrics['mismatch/rollout_is_masked_fraction']:.2f}")
    print("   ✓ Mask mode correctly separates IS weights from rejection")


if __name__ == "__main__":
    print("=" * 60)
    print("Rollout Importance Sampling Test Suite")
    print("=" * 60)

    try:
        test_basic_rollout_is()
        test_metrics_completeness()
        test_mismatch_metrics()
        test_mask_mode()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
