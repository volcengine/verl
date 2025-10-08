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
        """Create config with rollout IS enabled."""
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size=2,
            rollout_is=True,
            rollout_is_threshold=2.0,
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_veto_threshold=1e-4,
            clip_ratio=0.2,
        )
        return config

    def test_policy_loss_with_rollout_is(self, sample_data, config_with_rollout_is):
        """Test that policy loss computation works with rollout IS."""
        # Policy loss function applies IS correction internally
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_log_probs=sample_data["rollout_log_prob"],
        )

        # Check loss is valid
        assert isinstance(pg_loss, torch.Tensor)
        assert pg_loss.ndim == 0  # Scalar
        assert not torch.isnan(pg_loss)
        assert not torch.isinf(pg_loss)

    def test_rollout_is_weights_computation(self, sample_data):
        """Test rollout IS weights and metrics computation."""
        weights, metrics = compute_rollout_importance_weights(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            eos_mask=sample_data["response_mask"],
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=2.0,
            rollout_is_veto_threshold=1e-4,
        )

        # Check weights
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == sample_data["old_log_prob"].shape

        # Check metrics are returned
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert "rollout_is_mean" in metrics
        assert "rollout_is_threshold_upper" in metrics

    def test_all_aggregation_levels(self, sample_data):
        """Test all three aggregation levels."""
        levels = ["token", "sequence", "geometric"]

        for level in levels:
            _, metrics = compute_rollout_importance_weights(
                old_log_prob=sample_data["old_log_prob"],
                rollout_log_prob=sample_data["rollout_log_prob"],
                eos_mask=sample_data["response_mask"],
                rollout_is_level=level,
                rollout_is_mode="truncate",
                rollout_is_threshold=2.0,
            )

            assert metrics["rollout_is_level"] == level
            assert "rollout_is_mean" in metrics

    def test_both_bounding_modes(self, sample_data):
        """Test both truncate and clip modes."""
        modes = ["truncate", "clip"]

        for mode in modes:
            _, metrics = compute_rollout_importance_weights(
                old_log_prob=sample_data["old_log_prob"],
                rollout_log_prob=sample_data["rollout_log_prob"],
                eos_mask=sample_data["response_mask"],
                rollout_is_level="token",
                rollout_is_mode=mode,
                rollout_is_threshold=2.0,
                rollout_is_threshold_lower=0.5,
            )

            assert metrics["rollout_is_mode"] == mode

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
            eos_mask=response_mask,
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=2.0,
            rollout_is_veto_threshold=1e-4,
        )

        # Should have vetoed one sequence
        assert metrics["rollout_is_veto_fraction"] > 0
        assert metrics["rollout_is_veto_fraction"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
