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
"""Integration tests for Rollout Importance Sampling."""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.mismatch_helper import compute_mismatch_metrics, compute_rollout_importance_weights
from verl.workers.config.actor import ActorConfig


class TestRolloutISIntegration:
    """Integration tests for Rollout IS with PPO."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        batch_size, seq_length = 4, 16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "old_log_prob": torch.randn(batch_size, seq_length, device=device),
            "log_prob": torch.randn(batch_size, seq_length, device=device),
            "rollout_log_prob": torch.randn(batch_size, seq_length, device=device),
            "advantages": torch.randn(batch_size, seq_length, device=device),
            "response_mask": torch.ones(batch_size, seq_length, device=device),
        }

    @pytest.fixture
    def config_with_rollout_is(self):
        """Create config for policy loss computation.

        Note: rollout_is config has been moved to algorithm config.
        This config only needs fields used by policy loss (clip_ratio, etc).
        """
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size=2,
            clip_ratio=0.2,
        )
        return config

    def test_policy_loss_with_rollout_is(self, sample_data, config_with_rollout_is):
        """Test that policy loss computation works with rollout IS weights.

        Note: In production, IS weights are computed centrally in the trainer
        (before advantage computation) and passed to policy loss.
        This test simulates that workflow.
        """
        # First compute IS weights (as trainer would do centrally)
        rollout_is_weights_proto, _ = compute_rollout_importance_weights(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=2.0,
            rollout_is_veto_threshold=1e-4,
        )

        rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"]

        # Policy loss function receives pre-computed IS weights
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=rollout_is_weights,
        )

        # Check loss is valid
        assert isinstance(pg_loss, torch.Tensor)
        assert pg_loss.ndim == 0  # Scalar
        assert not torch.isnan(pg_loss)
        assert not torch.isinf(pg_loss)

    def test_rollout_is_weights_computation(self, sample_data):
        """Test rollout IS weights and metrics computation."""
        weights_proto, metrics = compute_rollout_importance_weights(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=2.0,
            rollout_is_veto_threshold=1e-4,
        )

        # Check weights
        from verl.protocol import DataProto

        assert isinstance(weights_proto, DataProto)
        weights = weights_proto.batch["rollout_is_weights"]
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == sample_data["old_log_prob"].shape

        # Check metrics are returned
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert "mismatch/rollout_is_mean" in metrics

    def test_all_aggregation_levels(self, sample_data):
        """Test all three aggregation levels."""
        levels = ["token", "sequence", "geometric"]

        for level in levels:
            _, metrics = compute_rollout_importance_weights(
                old_log_prob=sample_data["old_log_prob"],
                rollout_log_prob=sample_data["rollout_log_prob"],
                response_mask=sample_data["response_mask"],
                rollout_is_level=level,
                rollout_is_mode="truncate",
                rollout_is_threshold=2.0,
            )

            assert "mismatch/rollout_is_mean" in metrics

    def test_both_bounding_modes(self, sample_data):
        """Test both truncate and mask modes."""
        modes = ["truncate", "mask"]

        for mode in modes:
            _, metrics = compute_rollout_importance_weights(
                old_log_prob=sample_data["old_log_prob"],
                rollout_log_prob=sample_data["rollout_log_prob"],
                response_mask=sample_data["response_mask"],
                rollout_is_level="token",
                rollout_is_mode=mode,
                rollout_is_threshold=2.0,
                rollout_is_threshold_lower=0.5,
            )

            assert "mismatch/rollout_is_mean" in metrics

    def test_mismatch_metrics(self, sample_data):
        """Test mismatch diagnostic metrics computation."""
        metrics = compute_mismatch_metrics(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
        )

        # Check key metrics are present
        assert "mismatch_training_ppl" in metrics
        assert "mismatch_rollout_ppl" in metrics
        assert "mismatch_kl" in metrics
        assert isinstance(metrics["mismatch_kl"], float)

    def test_veto_mechanism(self):
        """Test veto mechanism with catastrophic outliers."""
        batch_size, seq_length = 2, 5
        device = "cuda" if torch.cuda.is_available() else "cpu"

        old_log_prob = torch.randn(batch_size, seq_length, device=device)
        rollout_log_prob = old_log_prob.clone()

        # Create catastrophic outlier in first sequence
        rollout_log_prob[0, 2] += 15.0  # Makes ratio ~3e-7

        response_mask = torch.ones(batch_size, seq_length, device=device)

        _, metrics = compute_rollout_importance_weights(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_prob,
            response_mask=response_mask,
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=2.0,
            rollout_is_veto_threshold=1e-4,
        )

        # Should have vetoed one sequence
        assert metrics["mismatch/rollout_is_veto_fraction"] > 0
        assert metrics["mismatch/rollout_is_veto_fraction"] <= 1.0

    def test_metrics_only_mode(self, sample_data, config_with_rollout_is):
        """Test metrics-only mode: compute IS weights/metrics but don't apply to loss.

        This tests the use case where rollout_is_threshold is set (enables computation)
        but rollout_is=False (disables weight application to policy loss).
        """
        # Compute IS weights (as trainer would do)
        rollout_is_weights_proto, is_metrics = compute_rollout_importance_weights(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=2.0,
        )

        # Metrics should be computed
        assert len(is_metrics) > 0
        assert "mismatch/rollout_is_mean" in is_metrics

        # In metrics-only mode, we compute loss WITHOUT applying weights
        # (simulating rollout_is=False)
        pg_loss_no_weights, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=None,  # Don't apply weights
        )

        # Compare to loss WITH weights (rollout_is=True)
        rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"]
        pg_loss_with_weights, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=rollout_is_weights,
        )

        # Losses should be different (weights have an effect)
        assert not torch.allclose(pg_loss_no_weights, pg_loss_with_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
