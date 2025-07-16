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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.trainer.config.config import CriticConfig, FSDPCriticConfig, MegatronCriticConfig
from verl.utils.config import omega_conf_to_dataclass


class TestCriticConfig:
    """Test suite for critic configuration dataclasses."""

    @pytest.fixture
    def config_dir(self):
        """Get the path to the config directory."""
        return Path(__file__).parent.parent.parent.parent / "verl" / "trainer" / "config" / "critic"

    def test_megatron_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that MegatronCriticConfig can be instantiated from megatron_critic.yaml."""
        yaml_path = config_dir / "megatron_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        from hydra import compose, initialize_config_dir
        import os
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/critic")):
            test_config = compose(config_name="megatron_critic")

        megatron_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(megatron_config_obj, MegatronCriticConfig)

        assert isinstance(megatron_config_obj, CriticConfig)

        assert hasattr(megatron_config_obj, "strategy")
        assert hasattr(megatron_config_obj, "rollout_n")
        assert hasattr(megatron_config_obj, "optim")
        assert hasattr(megatron_config_obj, "model")
        assert hasattr(megatron_config_obj, "ppo_mini_batch_size")
        assert hasattr(megatron_config_obj, "ppo_max_token_len_per_gpu")
        assert hasattr(megatron_config_obj, "cliprange_value")

        assert hasattr(megatron_config_obj, "get")
        assert callable(megatron_config_obj.get)

        assert hasattr(megatron_config_obj, "nccl_timeout")
        assert hasattr(megatron_config_obj, "megatron")
        assert hasattr(megatron_config_obj, "load_weight")


        assert megatron_config_obj.strategy == "megatron"

    def test_fsdp_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that FSDPCriticConfig can be instantiated from dp_critic.yaml."""
        yaml_path = config_dir / "dp_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        from hydra import compose, initialize_config_dir
        import os
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/critic")):
            test_config = compose(config_name="dp_critic")

        fsdp_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(fsdp_config_obj, FSDPCriticConfig)

        assert isinstance(fsdp_config_obj, CriticConfig)

        assert hasattr(fsdp_config_obj, "strategy")
        assert hasattr(fsdp_config_obj, "rollout_n")
        assert hasattr(fsdp_config_obj, "optim")
        assert hasattr(fsdp_config_obj, "model")
        assert hasattr(fsdp_config_obj, "ppo_mini_batch_size")
        assert hasattr(fsdp_config_obj, "ppo_max_token_len_per_gpu")
        assert hasattr(fsdp_config_obj, "cliprange_value")

        assert hasattr(fsdp_config_obj, "get")
        assert callable(fsdp_config_obj.get)

        assert hasattr(fsdp_config_obj, "forward_micro_batch_size")
        assert hasattr(fsdp_config_obj, "forward_micro_batch_size_per_gpu")
        assert hasattr(fsdp_config_obj, "ulysses_sequence_parallel_size")
        assert hasattr(fsdp_config_obj, "grad_clip")

        assert fsdp_config_obj.strategy == "fsdp"

    def test_config_inheritance_hierarchy(self):
        """Test that the inheritance hierarchy is correct."""
        megatron_config = MegatronCriticConfig()
        assert isinstance(megatron_config, CriticConfig)
        assert isinstance(megatron_config, MegatronCriticConfig)

        fsdp_config = FSDPCriticConfig()
        assert isinstance(fsdp_config, CriticConfig)
        assert isinstance(fsdp_config, FSDPCriticConfig)

        critic_config = CriticConfig()
        assert isinstance(critic_config, CriticConfig)
        assert not isinstance(critic_config, MegatronCriticConfig)
        assert not isinstance(critic_config, FSDPCriticConfig)

    def test_config_dict_interface(self):
        """Test that configs provide dict-like interface from BaseConfig."""
        config = CriticConfig()

        assert "strategy" in config
        assert config["strategy"] == "fsdp"

        assert config.get("strategy") == "fsdp"
        assert config.get("nonexistent_key", "default") == "default"

        keys = list(config)
        assert "strategy" in keys
        assert "rollout_n" in keys

        assert len(config) > 0
