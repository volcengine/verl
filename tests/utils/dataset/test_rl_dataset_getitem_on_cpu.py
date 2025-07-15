# Copyright 2025 Amazon.com Inc and/or its affiliates
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

from unittest.mock import Mock

import datasets
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


def create_test_dataset(data_dict, tokenizer, config, processor=None):
    """Helper to create RLHFDataset with controlled dataframe"""
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Create dataset without calling __init__
    dataset = RLHFDataset.__new__(RLHFDataset)

    # Set required attributes
    dataset.dataframe = datasets.Dataset.from_dict(data_dict)
    dataset.tokenizer = tokenizer
    dataset.processor = processor
    dataset.config = config
    dataset.prompt_key = config.get("prompt_key", "prompt")
    dataset.image_key = config.get("image_key", "images")
    dataset.video_key = config.get("video_key", "videos")
    dataset.max_prompt_length = config.get("max_prompt_length", 1024)
    dataset.return_raw_chat = config.get("return_raw_chat", False)
    dataset.return_full_prompt = config.get("return_full_prompt", False)
    dataset.truncation = config.get("truncation", "error")
    dataset.need_tools_kwargs = config.get("need_tools_kwargs", False)
    dataset.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

    return dataset


class TestRLHFDatasetGetItem:
    @pytest.fixture
    def tokenizer(self):
        from verl.utils import hf_tokenizer

        return hf_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")

    @pytest.fixture
    def basic_config(self):
        return OmegaConf.create(
            {
                "prompt_key": "prompt",
                "max_prompt_length": 128,
                "truncation": "error",
                "return_raw_chat": False,
                "return_full_prompt": False,
                "need_tools_kwargs": False,
            }
        )

    def test_basic_text_only_item(self, tokenizer, basic_config):
        """Test basic text-only prompt processing"""
        data_dict = {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "What is 2+2?",
                    }
                ]
            ],
            "extra_info": [{"index": 0}],
        }

        dataset = create_test_dataset(data_dict, tokenizer, basic_config)

        result = dataset[0]

        # Check required fields are present
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "position_ids" in result
        assert "raw_prompt_ids" in result
        assert "index" in result
        assert "tools_kwargs" in result
        assert "interaction_kwargs" in result

        # Check shapes
        assert result["input_ids"].shape == torch.Size([128])
        assert result["attention_mask"].shape == torch.Size([128])
        assert result["position_ids"].shape == torch.Size([128])

        # Check index
        assert result["index"] == 0

    def test_with_extra_info_fields(self, tokenizer, basic_config):
        """Test handling of extra_info with tools_kwargs and interaction_kwargs"""
        data_dict = {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "extra_info": [
                {
                    "index": 42,
                    "tools_kwargs": {"tool": "calculator"},
                    "interaction_kwargs": {"mode": "chat"},
                    "need_tools_kwargs": True,
                }
            ],
        }

        dataset = create_test_dataset(data_dict, tokenizer, basic_config)

        result = dataset[0]

        assert result["index"] == 42
        assert result["tools_kwargs"] == {"tool": "calculator"}
        assert result["interaction_kwargs"] == {"mode": "chat"}

    def test_return_raw_chat_enabled(self, tokenizer, basic_config):
        """Test return_raw_chat option"""
        config = basic_config.copy()
        config.return_raw_chat = True

        data_dict = {"prompt": [[{"role": "user", "content": "Test message"}]], "extra_info": [{"index": 0}]}

        dataset = create_test_dataset(data_dict, tokenizer, config)

        result = dataset[0]

        assert "raw_prompt" in result
        assert result["raw_prompt"] == [{"role": "user", "content": "Test message"}]

    def test_return_full_prompt_enabled(self, tokenizer, basic_config):
        """Test return_full_prompt option"""
        config = basic_config.copy()
        config.return_full_prompt = True

        data_dict = {"prompt": [[{"role": "user", "content": "Test"}]], "extra_info": [{"index": 0}]}

        dataset = create_test_dataset(data_dict, tokenizer, config)

        result = dataset[0]

        assert "full_prompts" in result
        assert isinstance(result["full_prompts"], str)

    def test_truncation_strategies(self, tokenizer, basic_config):
        """Test different truncation strategies"""
        # Test with a very small max_prompt_length to force truncation
        config = basic_config.copy()
        config.max_prompt_length = 5

        data_dict = {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "This is a very long message that will definitely exceed the token limit",
                    }
                ]
            ],
            "extra_info": [{"index": 0}],
        }

        # Test left truncation
        config.truncation = "left"
        dataset = create_test_dataset(data_dict, tokenizer, config)

        result = dataset[0]
        assert len(result["raw_prompt_ids"]) <= 5

    def test_missing_extra_info(self, tokenizer, basic_config):
        """Test handling when extra_info is missing or incomplete"""
        data_dict = {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            # No extra_info field
        }

        dataset = create_test_dataset(data_dict, tokenizer, basic_config)

        result = dataset[0]

        # Should have default values
        assert result["index"] == 0
        assert result["tools_kwargs"] == {}
        assert result["interaction_kwargs"] == {}

    def test_multimodal_with_images(self, tokenizer, basic_config):
        """Test multimodal processing with images"""

        # Create a fake PIL image
        fake_image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

        data_dict = {
            "prompt": [[{"role": "user", "content": "Describe this <image>"}]],
            "images": [[fake_image]],
            "extra_info": [{"index": 0}],
        }

        # Mock processor
        mock_processor = Mock()
        mock_processor.apply_chat_template.return_value = "Describe this image"
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.tensor([[[1.0]]]),
        }
        mock_processor.image_processor.__class__.__name__ = "SomeImageProcessor"

        dataset = create_test_dataset(data_dict, tokenizer, basic_config, processor=mock_processor)

        result = dataset[0]

        assert "multi_modal_data" in result
        assert "multi_modal_inputs" in result
        assert "image" in result["multi_modal_data"]

    def test_qwen2vl_position_ids(self, tokenizer, basic_config):
        """Test special position ID handling for Qwen2VL"""
        data_dict = {"prompt": [[{"role": "user", "content": "Test"}]], "extra_info": [{"index": 0}]}

        # Mock Qwen2VL processor
        mock_processor = Mock()
        mock_processor.apply_chat_template.return_value = "Test"
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "image_grid_thw": torch.tensor([1, 1, 1]),
        }
        mock_processor.image_processor.__class__.__name__ = "Qwen2VLImageProcessor"
        mock_processor.image_processor.merge_size = 2
        mock_processor.tokenizer.convert_tokens_to_ids.side_effect = lambda token: {
            "<|image_pad|>": 100,
            "<|video_pad|>": 101,
            "<|vision_start|>": 102,
        }.get(token, 0)

        dataset = create_test_dataset(data_dict, tokenizer, basic_config, processor=mock_processor)

        _ = dataset[0]

    def test_multi_modal_inputs_only_pops_second_per_grid_ts(self, tokenizer, basic_config):
        """Test that only 'second_per_grid_ts' is popped from multi_modal_inputs when return_multi_modal_inputs is True"""

        # Enable return_multi_modal_inputs
        config = basic_config.copy()
        config.return_multi_modal_inputs = True

        # Create a fake PIL image
        fake_image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

        data_dict = {
            "prompt": [[{"role": "user", "content": "Describe this <image>"}]],
            "images": [[fake_image]],
            "extra_info": [{"index": 0}],
        }

        # Mock processor that returns multi_modal_inputs with various keys including second_per_grid_ts
        mock_processor = Mock()
        mock_processor.apply_chat_template.return_value = "Describe this image"

        # Create mock multi_modal_inputs with multiple keys including second_per_grid_ts
        mock_multi_modal_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.tensor([[[1.0]]]),
            "second_per_grid_ts": torch.tensor([1.0, 2.0]),  # This should be popped
            "other_key": torch.tensor([5.0]),  # This should remain
            "another_key": "some_value",  # This should remain
        }

        mock_processor.return_value = mock_multi_modal_inputs
        mock_processor.image_processor.__class__.__name__ = "SomeImageProcessor"

        dataset = create_test_dataset(data_dict, tokenizer, config, processor=mock_processor)

        result = dataset[0]

        # Check that multi_modal_inputs is present in result
        assert "multi_modal_inputs" in result

        # Check that second_per_grid_ts was popped (not present in result)
        assert "second_per_grid_ts" not in result["multi_modal_inputs"]

        # Check that all other keys remain in multi_modal_inputs
        assert "input_ids" in result["multi_modal_inputs"]
        assert "attention_mask" in result["multi_modal_inputs"]
        assert "pixel_values" in result["multi_modal_inputs"]
        assert "other_key" in result["multi_modal_inputs"]
        assert "another_key" in result["multi_modal_inputs"]

        # Verify the values are correct for remaining keys
        assert torch.equal(result["multi_modal_inputs"]["other_key"], torch.tensor([5.0]))
        assert result["multi_modal_inputs"]["another_key"] == "some_value"
