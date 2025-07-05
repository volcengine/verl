# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.config import AlgoConfig, KLControlConfig, PFPPOConfig
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    get_adv_estimator_fn,
)
from verl.utils.config import omega_conf_to_dataclass


class TestAlgoConfig(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample algorithm config as DictConfig (similar to what comes from YAML)
        self.config_dict = {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 0.99,
            "lam": 0.95,
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "adaptive",
                "kl_coef": 0.002,
                "horizon": 5000,
                "target_kl": 0.05,
            },
            "use_pf_ppo": True,
            "pf_ppo": {"_target_": "verl.trainer.config.PFPPOConfig", "reweight_method": "max_min", "weight_pow": 3.0},
        }
        self.omega_config = OmegaConf.create(self.config_dict)

    def test_dataclass_creation_from_dict(self):
        """Test creating AlgoConfig from dictionary."""
        config = omega_conf_to_dataclass(self.config_dict)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)
        self.assertEqual(config.adv_estimator, "gae")
        self.assertTrue(config.norm_adv_by_std_in_grpo)
        self.assertTrue(config.use_kl_in_reward)
        self.assertEqual(config.kl_penalty, "kl")
        self.assertTrue(config.use_pf_ppo)

    def test_dataclass_creation_from_omega_config(self):
        """Test creating AlgoConfig from OmegaConf DictConfig."""
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)

    def test_nested_configs(self):
        """Test that nested configurations are properly converted."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test KL control config
        self.assertIsInstance(config.kl_ctrl, KLControlConfig)
        self.assertEqual(config.kl_ctrl.type, "adaptive")
        self.assertEqual(config.kl_ctrl.kl_coef, 0.002)
        self.assertEqual(config.kl_ctrl.horizon, 5000)
        self.assertEqual(config.kl_ctrl.target_kl, 0.05)

        # Test PF PPO config
        self.assertIsInstance(config.pf_ppo, PFPPOConfig)
        self.assertEqual(config.pf_ppo.reweight_method, "max_min")
        self.assertEqual(config.pf_ppo.weight_pow, 3.0)

    def test_default_values(self):
        """Test that default values are properly set."""
        minimal_config = {"gamma": 0.8}
        config = omega_conf_to_dataclass(minimal_config, AlgoConfig)

        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.lam, 1.0)  # default value
        self.assertEqual(config.adv_estimator, "gae")  # default value
        self.assertTrue(config.norm_adv_by_std_in_grpo)  # default value
        self.assertFalse(config.use_kl_in_reward)  # default value
        self.assertEqual(config.kl_penalty, "kl")  # default value
        self.assertFalse(config.use_pf_ppo)  # default value

    def test_get_method_backward_compatibility(self):
        """Test the get method for backward compatibility."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test existing attribute
        self.assertEqual(config.get("gamma"), 0.99)
        self.assertEqual(config.get("gamma", 1.0), 0.99)

        # Test non-existing attribute
        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_post_init_nested_configs(self):
        """Test that __post_init__ properly initializes nested configs when None."""
        # Create config without nested configs
        minimal_config = AlgoConfig(gamma=0.9)

        # Check that nested configs are initialized
        self.assertIsNotNone(minimal_config.kl_ctrl)
        self.assertIsInstance(minimal_config.kl_ctrl, KLControlConfig)
        self.assertIsNone(minimal_config.pf_ppo)

    def test_config_init_from_yaml(self):
        cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
        algo_config = omega_conf_to_dataclass(cfg.algorithm)
        from verl.trainer.config import AlgoConfig, PFPPOConfig

        assert isinstance(algo_config, AlgoConfig)
        assert isinstance(algo_config.pf_ppo, PFPPOConfig)


class TestAlgoCompute(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.algo_config = AlgoConfig(
            gamma=0.99,
            lam=0.95,
            adv_estimator="gae",
            norm_adv_by_std_in_grpo=True,
            use_kl_in_reward=True,
            kl_penalty="kl",
            kl_ctrl=KLControlConfig(type="adaptive", kl_coef=0.002, horizon=5000, target_kl=0.05),
            use_pf_ppo=True,
            pf_ppo=PFPPOConfig(reweight_method="max_min", weight_pow=3.0),
        )

    def test_advantage_estimator_with_cfg(self):
        """Test integration with advantage estimators from core_algos."""
        config = self.algo_config

        # Test GAE advantage estimator
        adv_fn = get_adv_estimator_fn(config.adv_estimator)
        self.assertIsNotNone(adv_fn)

        # Test with actual GAE computation
        batch_size, seq_len = 2, 5
        token_level_rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma,
            lam=config.lam,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))

    def test_grpo_advantage_estimator_with_cfg(self):
        """Test integration with GRPO advantage estimator."""
        grpo_config = AlgoConfig(adv_estimator="grpo", norm_adv_by_std_in_grpo=True)

        # Test GRPO advantage computation
        batch_size, seq_len = 4, 3
        token_level_rewards = torch.tensor([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [0.5, 0.2, 0.0], [1.5, 0.8, 0.0]])
        response_mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 1, 1])  # Two groups

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=grpo_config.norm_adv_by_std_in_grpo,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))


class TestConfigComparison(unittest.TestCase):
    """Test that current configs match their legacy counterparts exactly."""

    def _compare_configs_recursively(self, current_config, legacy_config, path=""):
        """Recursively compare two OmegaConf configs and assert they are identical."""
        if isinstance(current_config, dict) and isinstance(legacy_config, dict):
            current_keys = set(current_config.keys())
            legacy_keys = set(legacy_config.keys())

            missing_in_current = legacy_keys - current_keys
            missing_in_legacy = current_keys - legacy_keys

            if missing_in_current:
                self.fail(f"Keys missing in current config at {path}: {missing_in_current}")
            if missing_in_legacy:
                self.fail(f"Keys missing in legacy config at {path}: {missing_in_legacy}")

            for key in current_keys:
                current_path = f"{path}.{key}" if path else key
                self._compare_configs_recursively(current_config[key], legacy_config[key], current_path)
        elif isinstance(current_config, list) and isinstance(legacy_config, list):
            self.assertEqual(
                len(current_config),
                len(legacy_config),
                f"List lengths differ at {path}: current={len(current_config)}, legacy={len(legacy_config)}",
            )
            for i, (current_item, legacy_item) in enumerate(zip(current_config, legacy_config)):
                self._compare_configs_recursively(current_item, legacy_item, f"{path}[{i}]")
        else:
            self.assertEqual(
                current_config,
                legacy_config,
                f"Values differ at {path}: current={current_config}, legacy={legacy_config}",
            )

    def test_ppo_trainer_config_matches_legacy(self):
        """Test that ppo_trainer.yaml matches legacy_ppo_trainer.yaml exactly."""
        import os

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
                current_config = compose(config_name="ppo_trainer")

            legacy_config = OmegaConf.load("tests/trainer/config/legacy_ppo_trainer.yaml")

            current_dict = OmegaConf.to_container(current_config, resolve=True)
            legacy_dict = OmegaConf.to_container(legacy_config, resolve=True)

            if "defaults" in current_dict:
                del current_dict["defaults"]

            self._compare_configs_recursively(current_dict, legacy_dict)
        finally:
            GlobalHydra.instance().clear()

    def test_ppo_megatron_trainer_config_matches_legacy(self):
        """Test that ppo_megatron_trainer.yaml matches legacy_ppo_megatron_trainer.yaml exactly."""
        import os

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
                current_config = compose(config_name="ppo_megatron_trainer")

            legacy_config = OmegaConf.load("tests/trainer/config/legacy_ppo_megatron_trainer.yaml")

            current_dict = OmegaConf.to_container(current_config, resolve=True)
            legacy_dict = OmegaConf.to_container(legacy_config, resolve=True)

            if "defaults" in current_dict:
                del current_dict["defaults"]

            self._compare_configs_recursively(current_dict, legacy_dict)
        finally:
            GlobalHydra.instance().clear()


if __name__ == "__main__":
    unittest.main()
