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
        return Path(__file__).parent.parent.parent / "verl" / "trainer" / "config" / "critic"

    def test_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that CriticConfig can be instantiated using _target_ field."""
        yaml_path = config_dir / "critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        config = OmegaConf.create(
            {
                "_target_": "verl.trainer.config.config.CriticConfig",
                "strategy": "fsdp",
                "rollout_n": 4,
                "optim": {"lr": 0.001},
                "model": {"path": "~/models/test-model"},
                "ppo_mini_batch_size": 2,
                "ppo_max_token_len_per_gpu": 32768,
                "cliprange_value": 0.5,
            }
        )

        critic_config = omega_conf_to_dataclass(config)

        assert isinstance(critic_config, CriticConfig)

        assert hasattr(critic_config, "strategy")
        assert hasattr(critic_config, "rollout_n")
        assert hasattr(critic_config, "optim")
        assert hasattr(critic_config, "model")
        assert hasattr(critic_config, "ppo_mini_batch_size")
        assert hasattr(critic_config, "ppo_max_token_len_per_gpu")
        assert hasattr(critic_config, "cliprange_value")

        assert hasattr(critic_config, "get")
        assert callable(critic_config.get)

    def test_megatron_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that MegatronCriticConfig can be instantiated using _target_ field."""
        yaml_path = config_dir / "megatron_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        config = OmegaConf.create(
            {
                "_target_": "verl.trainer.config.config.MegatronCriticConfig",
                "strategy": "megatron",
                "rollout_n": 4,
                "optim": {"lr": 0.001},
                "model": {"path": "~/models/test-model"},
                "ppo_mini_batch_size": 2,
                "ppo_max_token_len_per_gpu": 32768,
                "cliprange_value": 0.5,
                "nccl_timeout": 600,
                "megatron": {"seed": 42},
                "load_weight": True,
                "kl_ctrl": {},
            }
        )

        megatron_config_obj = omega_conf_to_dataclass(config)

        assert isinstance(megatron_config_obj, MegatronCriticConfig)

        assert isinstance(megatron_config_obj, CriticConfig)

        assert hasattr(megatron_config_obj, "nccl_timeout")
        assert hasattr(megatron_config_obj, "megatron")
        assert hasattr(megatron_config_obj, "load_weight")
        assert hasattr(megatron_config_obj, "kl_ctrl")

        assert megatron_config_obj.strategy == "megatron"

    def test_fsdp_critic_config_instantiation_from_yaml(self, config_dir):
        """Test that FSDPCriticConfig can be instantiated using _target_ field."""
        yaml_path = config_dir / "dp_critic.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        config = OmegaConf.create(
            {
                "_target_": "verl.trainer.config.config.FSDPCriticConfig",
                "strategy": "fsdp",
                "rollout_n": 4,
                "optim": {"lr": 0.001},
                "model": {"path": "~/models/test-model"},
                "ppo_mini_batch_size": 2,
                "ppo_max_token_len_per_gpu": 32768,
                "cliprange_value": 0.5,
                "forward_micro_batch_size": 1,
                "forward_micro_batch_size_per_gpu": 1,
                "ulysses_sequence_parallel_size": 1,
                "grad_clip": 1.0,
            }
        )

        fsdp_config_obj = omega_conf_to_dataclass(config)

        assert isinstance(fsdp_config_obj, FSDPCriticConfig)

        assert isinstance(fsdp_config_obj, CriticConfig)

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
