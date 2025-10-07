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
        result = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_log_probs=sample_data["rollout_log_prob"],
            return_rollout_is_metrics=True,
        )

        assert len(result) == 5, "Should return 5 values with metrics"
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, rollout_is_metrics = result

        # Check loss is valid
        assert isinstance(pg_loss, torch.Tensor)
        assert pg_loss.ndim == 0  # Scalar
        assert not torch.isnan(pg_loss)
        assert not torch.isinf(pg_loss)

        # Check metrics are returned
        assert isinstance(rollout_is_metrics, dict)
        assert len(rollout_is_metrics) > 0
        assert "rollout_is_mean" in rollout_is_metrics
        assert "rollout_is_threshold_upper" in rollout_is_metrics

    def test_policy_loss_without_rollout_is(self, sample_data):
        """Test that policy loss works without rollout IS."""
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            rollout_is=False,
            rollout_is_threshold=None,
            clip_ratio=0.2,
        )

        result = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config,
            rollout_log_probs=None,
        )

        assert len(result) == 4, "Should return 4 values without metrics"
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = result
        assert isinstance(pg_loss, torch.Tensor)

    def test_all_aggregation_levels(self, sample_data):
        """Test all three aggregation levels."""
        levels = ["token", "sequence", "geometric"]

        for level in levels:
            config = ActorConfig(
                strategy="fsdp",
                rollout_n=1,
                rollout_is=True,
                rollout_is_threshold=2.0,
                rollout_is_level=level,
                rollout_is_mode="truncate",
                clip_ratio=0.2,
            )

            result = compute_policy_loss_vanilla(
                old_log_prob=sample_data["old_log_prob"],
                log_prob=sample_data["log_prob"],
                advantages=sample_data["advantages"],
                response_mask=sample_data["response_mask"],
                loss_agg_mode="token-mean",
                config=config,
                rollout_log_probs=sample_data["rollout_log_prob"],
                return_rollout_is_metrics=True,
            )

            pg_loss, _, _, _, metrics = result
            assert not torch.isnan(pg_loss), f"Loss is NaN for level={level}"
            assert metrics["rollout_is_level"] == level

    def test_both_bounding_modes(self, sample_data):
        """Test both truncate and clip modes."""
        modes = ["truncate", "clip"]

        for mode in modes:
            config = ActorConfig(
                strategy="fsdp",
                rollout_n=1,
                rollout_is=True,
                rollout_is_threshold=2.0,
                rollout_is_threshold_lower=0.5,
                rollout_is_level="token",
                rollout_is_mode=mode,
                clip_ratio=0.2,
            )

            result = compute_policy_loss_vanilla(
                old_log_prob=sample_data["old_log_prob"],
                log_prob=sample_data["log_prob"],
                advantages=sample_data["advantages"],
                response_mask=sample_data["response_mask"],
                loss_agg_mode="token-mean",
                config=config,
                rollout_log_probs=sample_data["rollout_log_prob"],
                return_rollout_is_metrics=True,
            )

            pg_loss, _, _, _, metrics = result
            assert not torch.isnan(pg_loss), f"Loss is NaN for mode={mode}"
            assert metrics["rollout_is_mode"] == mode

    def test_metrics_completeness(self, sample_data, config_with_rollout_is):
        """Test that all expected metrics are present."""
        _, _, _, _, metrics = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_log_probs=sample_data["rollout_log_prob"],
            return_rollout_is_metrics=True,
        )

        expected_metrics = [
            "rollout_is_mean",
            "rollout_is_max",
            "rollout_is_min",
            "rollout_is_std",
            "rollout_is_eff_sample_size",
            "rollout_is_threshold_upper",
            "rollout_is_threshold_lower",
            "rollout_is_level",
            "rollout_is_mode",
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_veto_mechanism(self, config_with_rollout_is):
        """Test veto mechanism with catastrophic outliers."""
        batch_size, seq_length = 2, 5
        device = "cuda" if torch.cuda.is_available() else "cpu"

        old_log_prob = torch.randn(batch_size, seq_length, device=device)
        rollout_log_prob = old_log_prob.clone()

        # Create catastrophic outlier in first sequence
        rollout_log_prob[0, 2] += 15.0  # Makes ratio ~3e-7

        log_prob = torch.randn(batch_size, seq_length, device=device)
        advantages = torch.randn(batch_size, seq_length, device=device)
        response_mask = torch.ones(batch_size, seq_length, device=device)

        _, _, _, _, metrics = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_log_probs=rollout_log_prob,
            return_rollout_is_metrics=True,
        )

        # Should have vetoed one sequence
        assert metrics["rollout_is_veto_fraction"] > 0
        assert metrics["rollout_is_veto_fraction"] <= 1.0

    def test_gradient_flow(self, sample_data, config_with_rollout_is):
        """Test that gradients flow correctly through the loss."""
        sample_data["old_log_prob"].requires_grad_(True)
        sample_data["log_prob"].requires_grad_(True)

        pg_loss, _, _, _, _ = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_log_probs=sample_data["rollout_log_prob"],
            return_rollout_is_metrics=True,
        )

        pg_loss.backward()

        # Check gradients exist and are not NaN
        assert sample_data["log_prob"].grad is not None
        assert not torch.isnan(sample_data["log_prob"].grad).any()

    def test_dual_threshold_formats(self, sample_data):
        """Test both auto-reciprocal and explicit dual thresholds."""
        # Auto-reciprocal (lower = 1/upper)
        config_auto = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            rollout_is=True,
            rollout_is_threshold=2.0,
            rollout_is_threshold_lower=None,  # Auto
            rollout_is_level="token",
            rollout_is_mode="clip",
            clip_ratio=0.2,
        )

        _, _, _, _, metrics_auto = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_auto,
            rollout_log_probs=sample_data["rollout_log_prob"],
            return_rollout_is_metrics=True,
        )

        assert metrics_auto["rollout_is_threshold_lower"] == 0.5  # 1/2.0

        # Explicit dual thresholds
        config_explicit = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            rollout_is=True,
            rollout_is_threshold=3.0,
            rollout_is_threshold_lower=0.4,  # Explicit
            rollout_is_level="token",
            rollout_is_mode="clip",
            clip_ratio=0.2,
        )

        _, _, _, _, metrics_explicit = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_explicit,
            rollout_log_probs=sample_data["rollout_log_prob"],
            return_rollout_is_metrics=True,
        )

        assert metrics_explicit["rollout_is_threshold_lower"] == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
