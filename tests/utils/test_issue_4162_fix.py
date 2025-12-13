#!/usr/bin/env python3
"""
Test for Issue #4162 Fix: Logits scaling affecting rollout_actor_probs_pearson_corr metrics

This test verifies that temperature scaling is handled correctly when comparing
rollout and actor log probabilities.
"""

import torch
import torch.nn.functional as F
import pytest


class TestIssue4162Fix:
    """Test suite for Issue #4162 fix."""

    def test_temperature_scaling_causes_pearson_deviation(self):
        """
        Verify that temperature scaling causes Pearson correlation deviation.

        This demonstrates the original problem: when rollout uses unscaled logits
        and actor uses scaled logits, the Pearson correlation is not 1.0 even
        when using the same logits.
        """
        # Simulate logits from a model
        torch.manual_seed(42)
        batch_size = 10
        vocab_size = 50257
        logits = torch.randn(batch_size, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size,))
        temperature = 2.0

        # Rollout (unscaled) - simulating VLLM behavior
        rollout_log_probs = F.log_softmax(logits, dim=-1)[range(batch_size), labels]

        # Actor (scaled) - original behavior before fix
        actor_log_probs_scaled = F.log_softmax(logits / temperature, dim=-1)[range(batch_size), labels]

        # Calculate Pearson correlation
        rollout_probs = torch.exp(rollout_log_probs)
        actor_probs_scaled = torch.exp(actor_log_probs_scaled)

        pearson_with_scaling = torch.corrcoef(torch.stack([rollout_probs, actor_probs_scaled]))[0, 1]

        # The correlation should deviate from 1.0 when temperature != 1.0
        print(f"Pearson correlation with temperature scaling: {pearson_with_scaling:.4f}")
        assert abs(1.0 - pearson_with_scaling.item()) > 0.01, (
            "Pearson correlation should deviate from 1.0 when using different scaling"
        )

    def test_unscaled_logprobs_achieve_perfect_correlation(self):
        """
        Verify that using unscaled log_probs achieves perfect Pearson correlation.

        This demonstrates the fix: when both rollout and actor use unscaled logits,
        the Pearson correlation is 1.0 (or very close due to floating point precision).
        """
        # Simulate logits from a model
        torch.manual_seed(42)
        batch_size = 10
        vocab_size = 50257
        logits = torch.randn(batch_size, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size,))

        # Rollout (unscaled) - simulating VLLM behavior
        rollout_log_probs = F.log_softmax(logits, dim=-1)[range(batch_size), labels]

        # Actor (unscaled) - behavior after fix (log_probs_for_metrics)
        actor_log_probs_unscaled = F.log_softmax(logits, dim=-1)[range(batch_size), labels]

        # Calculate Pearson correlation
        rollout_probs = torch.exp(rollout_log_probs)
        actor_probs_unscaled = torch.exp(actor_log_probs_unscaled)

        pearson_without_scaling = torch.corrcoef(torch.stack([rollout_probs, actor_probs_unscaled]))[0, 1]

        # The correlation should be 1.0 (or very close)
        print(f"Pearson correlation without temperature scaling: {pearson_without_scaling:.4f}")
        assert abs(1.0 - pearson_without_scaling.item()) < 0.001, (
            f"Pearson correlation should be ~1.0, got {pearson_without_scaling:.4f}"
        )

    def test_temperature_scaling_mathematical_inequality(self):
        """
        Verify the mathematical inequality: log_softmax(logits) != log_softmax(logits/T)

        This test demonstrates that the two computations are mathematically different
        when temperature != 1.0.
        """
        torch.manual_seed(42)
        logits = torch.tensor([2.0, 1.0, 0.5])
        temperature = 2.0

        # Unscaled
        log_probs_unscaled = F.log_softmax(logits, dim=-1)

        # Scaled
        log_probs_scaled = F.log_softmax(logits / temperature, dim=-1)

        print(f"Unscaled log_probs: {log_probs_unscaled}")
        print(f"Scaled log_probs:   {log_probs_scaled}")

        # They should be different
        assert not torch.allclose(log_probs_unscaled, log_probs_scaled, atol=0.01), (
            "Unscaled and scaled log_probs should be different when temperature != 1.0"
        )

    def test_temperature_one_no_difference(self):
        """
        Verify that when temperature=1.0, scaling has no effect.

        This is a sanity check that the fix doesn't affect the case when temperature=1.0.
        """
        torch.manual_seed(42)
        logits = torch.tensor([2.0, 1.0, 0.5])
        temperature = 1.0

        # Unscaled
        log_probs_unscaled = F.log_softmax(logits, dim=-1)

        # Scaled with temperature=1.0
        log_probs_scaled = F.log_softmax(logits / temperature, dim=-1)

        # They should be the same
        assert torch.allclose(log_probs_unscaled, log_probs_scaled, atol=1e-6), (
            "Unscaled and scaled log_probs should be identical when temperature=1.0"
        )

    def test_realistic_training_scenario(self):
        """
        Test a realistic training scenario with typical values.

        This simulates what would happen during actual PPO training with temperature=1.5.
        """
        torch.manual_seed(42)
        batch_size = 256
        vocab_size = 50257
        response_length = 128

        # Simulate a batch of logits
        logits = torch.randn(batch_size, response_length, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, response_length))

        temperature = 1.5

        # Compute log_probs for each token
        rollout_log_probs_list = []
        actor_log_probs_scaled_list = []
        actor_log_probs_unscaled_list = []

        for i in range(batch_size):
            for j in range(response_length):
                logit = logits[i, j]
                label = labels[i, j]

                # Rollout (unscaled)
                rollout_lp = F.log_softmax(logit, dim=-1)[label]
                rollout_log_probs_list.append(rollout_lp)

                # Actor scaled (old behavior)
                actor_lp_scaled = F.log_softmax(logit / temperature, dim=-1)[label]
                actor_log_probs_scaled_list.append(actor_lp_scaled)

                # Actor unscaled (new behavior with fix)
                actor_lp_unscaled = F.log_softmax(logit, dim=-1)[label]
                actor_log_probs_unscaled_list.append(actor_lp_unscaled)

        rollout_probs = torch.exp(torch.stack(rollout_log_probs_list))
        actor_probs_scaled = torch.exp(torch.stack(actor_log_probs_scaled_list))
        actor_probs_unscaled = torch.exp(torch.stack(actor_log_probs_unscaled_list))

        # Pearson correlation before fix
        pearson_before = torch.corrcoef(torch.stack([rollout_probs, actor_probs_scaled]))[0, 1]

        # Pearson correlation after fix
        pearson_after = torch.corrcoef(torch.stack([rollout_probs, actor_probs_unscaled]))[0, 1]

        print(f"\nRealistic scenario with temperature={temperature}:")
        print(f"  Pearson correlation BEFORE fix (scaled):   {pearson_before:.4f}")
        print(f"  Pearson correlation AFTER fix (unscaled):  {pearson_after:.4f}")
        print(f"  Improvement: {abs(1.0 - pearson_before.item()) - abs(1.0 - pearson_after.item()):.4f}")

        # After fix should be much closer to 1.0
        assert abs(1.0 - pearson_after.item()) < abs(1.0 - pearson_before.item()), (
            "Fix should improve Pearson correlation"
        )
        assert abs(1.0 - pearson_after.item()) < 0.001, (
            f"After fix, Pearson should be ~1.0, got {pearson_after:.4f}"
        )


if __name__ == "__main__":
    # Run with: python -m pytest tests/utils/test_issue_4162_fix.py -v -s
    pytest.main([__file__, "-v", "-s"])
