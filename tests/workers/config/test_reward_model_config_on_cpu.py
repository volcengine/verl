# tests/workers/config/test_reward_model_config_on_cpu.py

import os
import unittest
from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RewardModelConfig

class TestRewardModelConfig(unittest.TestCase):
    """
    Test the RewardModelConfig dataclass, using 'ppo_trainer_rm.yaml' as reference.
    """

    @classmethod
    def setUpClass(cls):
        """Locate the example config directory."""
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
        cls.config_dir = os.path.join(project_root, "examples", "reward_model", "config")
        if not os.path.isdir(cls.config_dir):
            raise FileNotFoundError(f"Config directory not found at: {cls.config_dir}")

    def _get_base_overrides(self):
        """
        Provides overrides to resolve interpolations and set a valid base state.
        """
        return [
            "reward_model.enable=True",
            "reward_model.use_dynamic_bsz=False",
            "reward_model.micro_batch_size_per_gpu=8",
            # Resolve other '${...}' interpolations
            "reward_model.forward_max_token_len_per_gpu=16384",
            "reward_model.model.input_tokenizer=/path/to/dummy_tokenizer",
            "reward_model.model.external_lib=null",
            "reward_model.model.use_fused_kernels=False",
            "reward_model.model.rollout.prompt_length=512",
            "reward_model.model.rollout.response_length=512",
            "reward_model.model.rollout.log_prob_use_dynamic_bsz=False",
            "reward_model.model.rollout.log_prob_max_token_len_per_gpu=16384",
        ]

    def test_loading_example_config_successfully(self):
        """Test that 'ppo_trainer_rm.yaml' with overrides for generator mode loads successfully."""
        try:
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=self._get_base_overrides())
            
            config = omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
            self.assertIsInstance(config, RewardModelConfig)
            self.assertTrue(config.enable)
            self.assertEqual(config.rm_mode, "generator")
        except Exception as e:
            self.fail(f"Loading 'ppo_trainer_rm.yaml' failed unexpectedly: {e}")

    def test_validation_fails_for_invalid_rm_mode(self):
        """Test that an unsupported rm_mode value raises a validation error."""
        overrides = self._get_base_overrides() + ["reward_model.rm_mode=some_invalid_mode"]
        with self.assertRaises(ValueError) as cm:
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=overrides)
                omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
        self.assertIn("Invalid rm_mode", str(cm.exception))
        
    def test_batch_size_validation_logic(self):
        """Test that providing both micro_batch_size and _per_gpu raises an error."""
        overrides = self._get_base_overrides() + ["reward_model.micro_batch_size=16"]
        with self.assertRaises(ValueError) as cm:
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=overrides)
                omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
        self.assertIn("You have set both", str(cm.exception))

    def test_discriminator_mode_is_valid_when_enabled(self):
        """Test that rm_mode='discriminator' is a valid configuration when enabled."""
        try:
            overrides = self._get_base_overrides() + ["reward_model.rm_mode=discriminator"]
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=overrides)
            
            config = omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
            self.assertEqual(config.rm_mode, "discriminator")
            self.assertTrue(config.enable)
        except Exception as e:
            self.fail(f"Configuring for discriminator mode failed unexpectedly: {e}")

    def test_config_is_valid_when_disabled_regardless_of_mode(self):
        """
        Test that if enable=False, the configuration is always valid,
        even for combinations that would otherwise be invalid.
        """
        # A combination that is invalid if enabled (missing batch size)
        overrides = [ov for ov in self._get_base_overrides() if not ov.startswith("reward_model.micro_batch_size")]
        overrides.append("reward_model.enable=false")
        
        try:
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=overrides)
            config = omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
            self.assertFalse(config.enable) # Verify the main switch is off
        except Exception as e:
            self.fail(f"Config should be valid when disabled, but it failed: {e}")

    def test_no_batch_size_error_when_dynamic_bsz_is_true(self):
        """
        Test that if use_dynamic_bsz=True, no batch size error is raised
        even if both micro_batch_size fields are null.
        """
        # Remove batch size settings and enable dynamic bsz
        overrides = [ov for ov in self._get_base_overrides() if not ov.startswith("reward_model.micro_batch_size")]
        overrides.append("reward_model.use_dynamic_bsz=true")
        
        try:
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=overrides)
            config = omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
            self.assertTrue(config.use_dynamic_bsz)
        except Exception as e:
            self.fail(f"Validation should pass with use_dynamic_bsz=True, but failed: {e}")

    def test_config_loads_with_missing_data_processor_info(self):
        """
        Test that the config loads successfully even if data_processer
        function names are null, as defaults should be applied internally.
        """
        overrides = self._get_base_overrides() + ["reward_model.model.data_processer.preprocess_fn_name=null"]
        try:
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                cfg = compose(config_name="ppo_trainer_rm", overrides=overrides)
            config = omega_conf_to_dataclass(cfg.reward_model, RewardModelConfig)
            # The config object should be created, and the field will be None
            self.assertIsNone(config.model.data_processer.preprocess_fn_name)
        except Exception as e:
            self.fail(f"Config should load with null processor function name, but failed: {e}")

if __name__ == "__main__":
    unittest.main()
