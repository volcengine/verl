# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import tempfile

import pytest
from omegaconf import OmegaConf

from verl.interactions.base import BaseInteraction
from verl.interactions.gsm8k_interaction import Gsm8kInteraction
from verl.interactions.utils.interaction_registry import (
    get_interaction_class,
    initialize_interactions_from_config,
)


class TestInteractionRegistry:
    def test_get_interaction_class(self):
        """Test getting interaction class by name."""
        # Test getting base interaction class
        base_cls = get_interaction_class("verl.interactions.base.BaseInteraction")
        assert base_cls == BaseInteraction

        # Test getting gsm8k interaction class
        gsm8k_cls = get_interaction_class("verl.interactions.gsm8k_interaction.Gsm8kInteraction")
        assert gsm8k_cls == Gsm8kInteraction

    def test_initialize_single_interaction_from_config(self):
        """Test initializing single interaction from config."""
        # Create temporary config file
        config_content = {
            "interaction": [
                {
                    "name": "test_gsm8k",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {},
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            # Check that interaction was created
            assert len(interaction_map) == 1
            assert "test_gsm8k" in interaction_map
            assert isinstance(interaction_map["test_gsm8k"], Gsm8kInteraction)
            assert interaction_map["test_gsm8k"].name == "test_gsm8k"
        finally:
            os.unlink(temp_config_path)

    def test_initialize_multiple_interactions_from_config(self):
        """Test initializing multiple interactions from config."""
        config_content = {
            "interaction": [
                {
                    "name": "gsm8k_solver",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {},
                },
                {
                    "name": "base_agent",
                    "class_name": "verl.interactions.base.BaseInteraction",
                    "config": {"custom_param": "test_value"},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            # Check that both interactions were created
            assert len(interaction_map) == 2
            assert "gsm8k_solver" in interaction_map
            assert "base_agent" in interaction_map

            # Check types
            assert isinstance(interaction_map["gsm8k_solver"], Gsm8kInteraction)
            assert isinstance(interaction_map["base_agent"], BaseInteraction)

            # Check names were injected
            assert interaction_map["gsm8k_solver"].name == "gsm8k_solver"
            assert interaction_map["base_agent"].name == "base_agent"

            # Check custom config was passed
            assert interaction_map["base_agent"].config.get("custom_param") == "test_value"
        finally:
            os.unlink(temp_config_path)

    def test_initialize_interaction_without_explicit_name(self):
        """Test that interaction name is derived from class name when not specified."""
        config_content = {
            "interaction": [{"class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction", "config": {}}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            # Check that interaction name was derived from class name
            assert len(interaction_map) == 1
            assert "gsm8k" in interaction_map  # Should be "gsm8k" after removing "interaction" suffix
            assert isinstance(interaction_map["gsm8k"], Gsm8kInteraction)
            assert interaction_map["gsm8k"].name == "gsm8k"
        finally:
            os.unlink(temp_config_path)

    def test_initialize_empty_config(self):
        """Test initializing from empty config."""
        config_content = {"interaction": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)
            assert len(interaction_map) == 0
        finally:
            os.unlink(temp_config_path)

    def test_invalid_class_name(self):
        """Test handling of invalid class name."""
        config_content = {
            "interaction": [{"name": "invalid", "class_name": "invalid.module.InvalidClass", "config": {}}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            with pytest.raises(ModuleNotFoundError):
                initialize_interactions_from_config(temp_config_path)
        finally:
            os.unlink(temp_config_path)

    def test_duplicate_interaction_names(self):
        """Test handling of duplicate interaction names."""
        config_content = {
            "interaction": [
                {"name": "duplicate", "class_name": "verl.interactions.base.BaseInteraction", "config": {}},
                {
                    "name": "duplicate",
                    "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                    "config": {},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            with pytest.raises(ValueError, match="Duplicate interaction name 'duplicate' found"):
                initialize_interactions_from_config(temp_config_path)
        finally:
            os.unlink(temp_config_path)

    def test_auto_name_generation_edge_cases(self):
        """Test automatic name generation for various class name patterns."""
        config_content = {
            "interaction": [
                {"class_name": "verl.interactions.base.BaseInteraction", "config": {}},
                {"class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction", "config": {}},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(config_content, f.name)
            temp_config_path = f.name

        try:
            interaction_map = initialize_interactions_from_config(temp_config_path)

            # Check that names were generated correctly
            assert len(interaction_map) == 2
            assert "base" in interaction_map  # BaseInteraction -> base
            assert "gsm8k" in interaction_map  # Gsm8kInteraction -> gsm8k
        finally:
            os.unlink(temp_config_path)
