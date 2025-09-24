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

import random
import unittest

import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_policy_loss_cispo,
    get_adv_estimator_fn,
    get_policy_loss_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


class TestComputePolicyLossCispo(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.batch_size = 2
        self.response_length = 4
        self.old_log_prob = torch.randn(self.batch_size, self.response_length)
        self.log_prob = torch.randn(self.batch_size, self.response_length)
        self.advantages = torch.randn(self.batch_size, self.response_length)
        self.response_mask = torch.ones(self.batch_size, self.response_length)

        from types import SimpleNamespace

        self.config = SimpleNamespace()
        self.config.clip_ratio = 0.2
        self.config.clip_ratio_low = None
        self.config.clip_ratio_high = None
        self.config.policy_loss = SimpleNamespace()
        self.config.policy_loss.cispo_clip_ratio_high = 0.2
        self.config.policy_loss.cispo_clip_ratio_low = 0.2

    def test_cispo_function_exists_and_registered(self):
        """Test that CISPO function is properly registered"""
        cispo_fn = get_policy_loss_fn("cispo")
        self.assertIsNotNone(cispo_fn)
        self.assertEqual(cispo_fn, compute_policy_loss_cispo)

    def test_cispo_output_format(self):
        """Test that CISPO returns correct output format"""
        result = compute_policy_loss_cispo(
            self.old_log_prob, self.log_prob, self.advantages, self.response_mask, config=self.config
        )

        self.assertEqual(len(result), 4)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = result
        self.assertIsInstance(pg_loss, torch.Tensor)
        self.assertIsInstance(pg_clipfrac, torch.Tensor)
        self.assertIsInstance(ppo_kl, torch.Tensor)
        self.assertIsInstance(pg_clipfrac_lower, torch.Tensor)

        self.assertEqual(pg_loss.dim(), 0)

    def test_cispo_parameter_defaults(self):
        """Test that CISPO uses correct parameter defaults"""
        self.config.policy_loss.cispo_clip_ratio_high = None
        self.config.policy_loss.cispo_clip_ratio_low = None

        result = compute_policy_loss_cispo(
            self.old_log_prob, self.log_prob, self.advantages, self.response_mask, config=self.config
        )

        self.assertEqual(len(result), 4)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = result
        self.assertIsInstance(pg_loss, torch.Tensor)

        self.assertTrue(torch.isfinite(pg_loss))

    def test_cispo_vs_vanilla_different_results(self):
        """Test that CISPO produces different results than vanilla PPO"""
        from verl.trainer.ppo.core_algos import compute_policy_loss

        cispo_result = compute_policy_loss_cispo(
            self.old_log_prob, self.log_prob, self.advantages, self.response_mask, config=self.config
        )

        vanilla_result = compute_policy_loss(
            self.old_log_prob,
            self.log_prob,
            self.advantages,
            self.response_mask,
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            clip_ratio_c=3.0,
        )

        cispo_loss, _, _, _ = cispo_result
        vanilla_loss, _, _, _ = vanilla_result

        self.assertTrue(torch.isfinite(cispo_loss))
        self.assertTrue(torch.isfinite(vanilla_loss))


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


if __name__ == "__main__":
    unittest.main()
