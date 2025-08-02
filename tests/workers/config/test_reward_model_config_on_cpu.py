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

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler import ProfilerConfig
from verl.workers.config import (
    FSDPRewardModelConfig,
    McoreRewardModelConfig,
    RewardModelConfig,
)


class TestRewardModelConfig:
    """Test suite for reward model configuration dataclasses."""

    @pytest.fixture
    def config_dir(self):
        """Get the path to the config directory."""
        return Path(__file__).parent.parent.parent.parent / "verl" / "trainer" / "config" / "reward_model"

    def test_megatron_reward_model_config_instantiation_from_yaml(self, config_dir):
        """Test that McoreRewardModelConfig can be instantiated from megatron_reward_model.yaml."""
        yaml_path = config_dir / "megatron_reward_model.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/reward_model")):
            test_config = compose(config_name="megatron_reward_model", overrides=["micro_batch_size_per_gpu=1"])

        megatron_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(megatron_config_obj, McoreRewardModelConfig)
        assert isinstance(megatron_config_obj, RewardModelConfig)

        expected_attrs = [
            "strategy",
            "model",
            "micro_batch_size_per_gpu",
            "forward_max_token_len_per_gpu",
            "reward_manager",
            "launch_reward_fn_async",
            "sandbox_fusion",
            "max_length",
            "nccl_timeout",
            "megatron",
            "load_weight",
        ]
        for attr in expected_attrs:
            assert hasattr(megatron_config_obj, attr), f"Missing attribute: {attr}"

        assert megatron_config_obj.strategy == "megatron"

    def test_fsdp_reward_model_config_instantiation_from_yaml(self, config_dir):
        """Test that FSDPRewardModelConfig can be instantiated from dp_reward_model.yaml."""
        yaml_path = config_dir / "dp_reward_model.yaml"
        assert yaml_path.exists(), f"Config file not found: {yaml_path}"

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/reward_model")):
            test_config = compose(config_name="dp_reward_model", overrides=["micro_batch_size_per_gpu=1"])

        fsdp_config_obj = omega_conf_to_dataclass(test_config)

        assert isinstance(fsdp_config_obj, FSDPRewardModelConfig)
        assert isinstance(fsdp_config_obj, RewardModelConfig)

        expected_attrs = [
            "strategy",
            "model",
            "micro_batch_size_per_gpu",
            "forward_max_token_len_per_gpu",
            "reward_manager",
            "launch_reward_fn_async",
            "sandbox_fusion",
            "max_length",
            "ulysses_sequence_parallel_size",
        ]
        for attr in expected_attrs:
            assert hasattr(fsdp_config_obj, attr), f"Missing attribute: {attr}"

        assert fsdp_config_obj.strategy == "fsdp"

    def test_config_inheritance_hierarchy(self):
        """Test that the inheritance hierarchy is correct."""
        megatron_config = McoreRewardModelConfig(micro_batch_size_per_gpu=1)
        assert isinstance(megatron_config, RewardModelConfig)
        assert isinstance(megatron_config, McoreRewardModelConfig)

        fsdp_config = FSDPRewardModelConfig(micro_batch_size_per_gpu=1)
        assert isinstance(fsdp_config, RewardModelConfig)
        assert isinstance(fsdp_config, FSDPRewardModelConfig)

        rm_config = RewardModelConfig(micro_batch_size_per_gpu=1, strategy="fsdp2")
        assert isinstance(rm_config, RewardModelConfig)
        assert not isinstance(rm_config, McoreRewardModelConfig)
        assert not isinstance(rm_config, FSDPRewardModelConfig)

    def test_config_dict_interface(self):
        """Test that configs provide dict-like interface from BaseConfig."""
        config = RewardModelConfig(micro_batch_size_per_gpu=1, strategy="fsdp2")

        assert "strategy" in config
        assert config["strategy"] == "fsdp2"

        assert config.get("strategy") == "fsdp2"
        assert config.get("nonexistent_key", "default") == "default"

        keys = list(config)
        assert "strategy" in keys
        assert "reward_manager" in keys

        assert len(config) > 0

    def test_frozen_fields_immutability(self):
        """Test that frozen fields raise exceptions when modified after creation."""
        rm_config = RewardModelConfig(micro_batch_size_per_gpu=1, strategy="fsdp2")
        frozen_fields = ["reward_manager", "strategy", "launch_reward_fn_async"]

        for field_name in frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(rm_config, field_name, "modified_value")

        megatron_config = McoreRewardModelConfig(micro_batch_size_per_gpu=1)
        megatron_frozen_fields = ["nccl_timeout", "load_weight"]

        for field_name in megatron_frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(megatron_config, field_name, "modified_value")

        fsdp_config = FSDPRewardModelConfig(micro_batch_size_per_gpu=1)
        fsdp_frozen_fields = ["ulysses_sequence_parallel_size"]

        for field_name in fsdp_frozen_fields:
            with pytest.raises((AttributeError, TypeError, ValueError)):
                setattr(fsdp_config, field_name, "modified_value")

    def test_batch_size_fields_modifiable(self):
        """Test that batch size fields can be modified after creation."""
        rm_config = RewardModelConfig(micro_batch_size_per_gpu=1, strategy="fsdp2")

        rm_config.micro_batch_size = 4
        rm_config.micro_batch_size_per_gpu = 2

        assert rm_config.micro_batch_size == 4
        assert rm_config.micro_batch_size_per_gpu == 2

    def test_profiler_config_type_validation(self):
        """Test that profiler field has correct type and validation."""
        rm_config = RewardModelConfig(micro_batch_size_per_gpu=1, strategy="fsdp2")
        assert isinstance(rm_config.profiler, ProfilerConfig)
        assert rm_config.profiler.discrete is False
        assert rm_config.profiler.all_ranks is False
        assert rm_config.profiler.ranks == []

        custom_profiler = ProfilerConfig(discrete=True, all_ranks=True, ranks=[0, 1])
        rm_config_custom = RewardModelConfig(profiler=custom_profiler, micro_batch_size_per_gpu=1, strategy="fsdp2")
        assert isinstance(rm_config_custom.profiler, ProfilerConfig)
        assert rm_config_custom.profiler.discrete is True
        assert rm_config_custom.profiler.all_ranks is True
        assert rm_config_custom.profiler.ranks == [0, 1]

        profiler1 = ProfilerConfig(discrete=True, ranks=[0, 1])
        profiler2 = ProfilerConfig(all_ranks=True, ranks=[1, 2])

        union_result = profiler1.union(profiler2)
        assert union_result.discrete is True
        assert union_result.all_ranks is True
        assert set(union_result.ranks) == {0, 1, 2}

        intersect_result = profiler1.intersect(profiler2)
        assert intersect_result.discrete is False
        assert intersect_result.all_ranks is False
        assert intersect_result.ranks == [1]

    def test_reward_model_config_validation_logic(self):
        """Test the __post_init__ validation logic for RewardModelConfig."""
        valid_config = RewardModelConfig(strategy="fsdp2", micro_batch_size_per_gpu=2, use_dynamic_bsz=False)
        assert valid_config.micro_batch_size_per_gpu == 2

        valid_config2 = RewardModelConfig(
            strategy="fsdp2",
            micro_batch_size_per_gpu=None,
            micro_batch_size=4,
            use_dynamic_bsz=False,
        )
        assert valid_config2.micro_batch_size == 4

        dynamic_config = RewardModelConfig(strategy="fsdp2", micro_batch_size_per_gpu=2, use_dynamic_bsz=True)
        assert dynamic_config.use_dynamic_bsz is True

        with pytest.raises(ValueError, match="You have set both.*micro_batch_size.*AND.*micro_batch_size_per_gpu"):
            RewardModelConfig(
                strategy="fsdp2",
                micro_batch_size=4,
                micro_batch_size_per_gpu=2,
                use_dynamic_bsz=False,
            )

        with pytest.raises(
            ValueError, match="Please set at least one of.*micro_batch_size.*or.*micro_batch_size_per_gpu"
        ):
            RewardModelConfig(
                strategy="fsdp2",
                micro_batch_size=None,
                micro_batch_size_per_gpu=None,
                use_dynamic_bsz=False,
            )

    def test_fsdp_sequence_parallelism_validation(self):
        """Test FSDP sequence parallelism validation in FSDPRewardModelConfig.__post_init__."""
        valid_config = FSDPRewardModelConfig(
            micro_batch_size_per_gpu=2,
            ulysses_sequence_parallel_size=2,
            model={"use_remove_padding": True},
        )
        assert valid_config.ulysses_sequence_parallel_size == 2

        with pytest.raises(
            ValueError, match="When using sequence parallelism for reward model, you must enable.*use_remove_padding"
        ):
            FSDPRewardModelConfig(
                micro_batch_size_per_gpu=2,
                ulysses_sequence_parallel_size=2,
                model={"use_remove_padding": False},
            )

        valid_config_no_sp = FSDPRewardModelConfig(
            micro_batch_size_per_gpu=2,
            ulysses_sequence_parallel_size=1,
            model={"use_remove_padding": False},
        )
        assert valid_config_no_sp.ulysses_sequence_parallel_size == 1
