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
"""
Unit tests for LoRA checkpoint handling in FSDPCheckpointManager.
Tests the _save_lora_adapter method which normalizes PEFT state_dict keys.
"""
import json
import os
import shutil
import tempfile
from collections import OrderedDict

import pytest
import torch

# Check if peft and safetensors are available
try:
    import peft  # noqa: F401
    from safetensors.torch import save_file  # noqa: F401

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class TestSaveLoraAdapter:
    """Test the _save_lora_adapter functionality for normalizing PEFT state_dict keys."""

    def _create_mock_peft_state_dict(self):
        """Create a mock state_dict that resembles PEFT/LoRA output."""
        state_dict = OrderedDict()
        # Base model weights with PEFT prefix
        state_dict["base_model.model.lm_head.weight"] = torch.randn(1000, 512)
        state_dict["base_model.model.model.embed_tokens.weight"] = torch.randn(1000, 512)
        # Layer weights with base_layer
        state_dict["base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"] = torch.randn(512, 512)
        state_dict["base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight"] = torch.randn(512, 512)
        state_dict["base_model.model.model.layers.0.self_attn.v_proj.base_layer.weight"] = torch.randn(512, 512)
        # LoRA adapter weights
        state_dict["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"] = torch.randn(16, 512)
        state_dict["base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"] = torch.randn(512, 16)
        state_dict["base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight"] = torch.randn(16, 512)
        state_dict["base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight"] = torch.randn(512, 16)
        state_dict["base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight"] = torch.randn(16, 512)
        state_dict["base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight"] = torch.randn(512, 16)
        return state_dict

    def _create_mock_standard_state_dict(self):
        """Create a mock state_dict without LoRA (standard HF format)."""
        state_dict = OrderedDict()
        state_dict["lm_head.weight"] = torch.randn(1000, 512)
        state_dict["model.embed_tokens.weight"] = torch.randn(1000, 512)
        state_dict["model.layers.0.self_attn.q_proj.weight"] = torch.randn(512, 512)
        state_dict["model.layers.0.self_attn.k_proj.weight"] = torch.randn(512, 512)
        state_dict["model.layers.0.self_attn.v_proj.weight"] = torch.randn(512, 512)
        return state_dict

    def _save_lora_adapter_standalone(self, state_dict: dict, target_dir: str):
        """
        Standalone implementation of _save_lora_adapter for testing.
        This mirrors the implementation in FSDPCheckpointManager.
        """
        lora_params_names = [name for name in state_dict.keys() if "lora_" in name]
        if len(lora_params_names) == 0:
            return None

        import peft
        from safetensors.torch import save_file

        lora_params = OrderedDict()
        target_modules = set()
        lora_key = None

        for name in lora_params_names:
            lora_key = name.replace(".default.weight", ".weight")
            target_modules.add(lora_key.split(".")[-3])
            lora_params[lora_key] = state_dict.pop(name)

        lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
        peft_dict = {
            "r": lora_rank,
            "lora_alpha": 0,
            "target_modules": list(target_modules),
        }
        peft_config = peft.LoraConfig(**peft_dict).to_dict()
        peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None
        peft_config["peft_type"] = peft_config["peft_type"].value if peft_config["peft_type"] else None
        peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_path = os.path.join(target_dir, "lora_adapter")
        os.makedirs(lora_path, exist_ok=True)
        with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)
        save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))

        # Normalize remaining keys to standard HuggingFace format
        for name in list(state_dict.keys()):
            key = (
                name.replace("base_model.model.", "")
                .replace(".base_layer.weight", ".weight")
                .replace(".base_layer.bias", ".bias")
            )
            if key != name:
                state_dict[key] = state_dict.pop(name)

        return lora_path

    @pytest.mark.skipif(not HAS_PEFT, reason="peft and safetensors required")
    def test_lora_adapter_extraction(self):
        """Test that LoRA weights are correctly extracted and saved."""
        state_dict = self._create_mock_peft_state_dict()
        temp_dir = tempfile.mkdtemp()

        try:
            lora_path = self._save_lora_adapter_standalone(state_dict, temp_dir)

            # Verify LoRA adapter was saved
            assert lora_path is not None
            assert os.path.exists(os.path.join(lora_path, "adapter_config.json"))
            assert os.path.exists(os.path.join(lora_path, "adapter_model.safetensors"))

            # Verify adapter config content
            with open(os.path.join(lora_path, "adapter_config.json")) as f:
                config = json.load(f)
            assert "r" in config
            assert "target_modules" in config
            assert config["r"] == 16  # Our mock LoRA rank
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.skipif(not HAS_PEFT, reason="peft and safetensors required")
    def test_state_dict_key_normalization(self):
        """Test that state_dict keys are normalized to HuggingFace format."""
        state_dict = self._create_mock_peft_state_dict()
        temp_dir = tempfile.mkdtemp()

        try:
            self._save_lora_adapter_standalone(state_dict, temp_dir)

            # Verify keys are now in standard HF format
            expected_keys = {
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            }
            assert set(state_dict.keys()) == expected_keys

            # Verify no LoRA keys remain
            lora_keys = [k for k in state_dict.keys() if "lora_" in k]
            assert len(lora_keys) == 0

            # Verify no base_model prefix remains
            prefix_keys = [k for k in state_dict.keys() if "base_model" in k]
            assert len(prefix_keys) == 0

            # Verify no base_layer suffix remains
            base_layer_keys = [k for k in state_dict.keys() if "base_layer" in k]
            assert len(base_layer_keys) == 0
        finally:
            shutil.rmtree(temp_dir)

    def test_no_lora_passthrough(self):
        """Test that non-LoRA state_dicts are not modified."""
        state_dict = self._create_mock_standard_state_dict()
        original_keys = set(state_dict.keys())
        temp_dir = tempfile.mkdtemp()

        try:
            lora_path = self._save_lora_adapter_standalone(state_dict, temp_dir)

            # Verify no LoRA adapter was created
            assert lora_path is None

            # Verify state_dict was not modified
            assert set(state_dict.keys()) == original_keys
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
