# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from unittest.mock import patch

import pytest

from verl.interactions.gsm8k_interaction import Gsm8kInteraction


class TestGsm8kInteraction:
    """Test cases for Gsm8kInteraction class."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.config = {"name": "gsm8k"}
        self.interaction = Gsm8kInteraction(self.config)

    def test_init(self):
        """Test Gsm8kInteraction initialization."""
        assert self.interaction._instance_dict == {}
        assert self.interaction.config == self.config
        assert self.interaction.name == "gsm8k"

    @pytest.mark.asyncio
    async def test_start_interaction_with_instance_id(self):
        """Test start_interaction with provided instance_id."""
        instance_id = "test_instance"
        ground_truth = "42"

        result_id = await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        assert result_id == instance_id
        assert instance_id in self.interaction._instance_dict
        assert self.interaction._instance_dict[instance_id]["response"] == ""
        assert self.interaction._instance_dict[instance_id]["ground_truth"] == ground_truth
        assert self.interaction._instance_dict[instance_id]["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_start_interaction_without_instance_id(self):
        """Test start_interaction without provided instance_id (auto-generated)."""
        ground_truth = "42"

        result_id = await self.interaction.start_interaction(ground_truth=ground_truth)

        assert result_id is not None
        assert len(result_id) == 36  # UUID4 length
        assert result_id in self.interaction._instance_dict
        assert self.interaction._instance_dict[result_id]["ground_truth"] == ground_truth

    @pytest.mark.asyncio
    async def test_start_interaction_without_ground_truth(self):
        """Test start_interaction without ground_truth parameter."""
        instance_id = "test_instance"

        result_id = await self.interaction.start_interaction(instance_id=instance_id)

        assert result_id == instance_id
        assert self.interaction._instance_dict[instance_id]["ground_truth"] is None

    @pytest.mark.asyncio
    async def test_generate_response_correct_answer_with_prefix(self):
        """Test generate_response with correct answer already having #### prefix."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "assistant", "content": "#### 42"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert reward == 1.0
        assert metadata == {}
        assert self.interaction._instance_dict[instance_id]["response"] == "#### 42"

    @pytest.mark.asyncio
    async def test_generate_response_correct_answer_without_prefix(self):
        """Test generate_response with correct answer missing #### prefix."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "assistant", "content": "42"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert reward == 1.0
        assert self.interaction._instance_dict[instance_id]["response"] == "42"

    @pytest.mark.asyncio
    async def test_generate_response_incorrect_answer(self):
        """Test generate_response with incorrect answer."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "assistant", "content": "24"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert response == "Your response is incorrect! You need to reflect on your answer and try again."
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == "24"

    @pytest.mark.asyncio
    async def test_generate_response_multiple_messages(self):
        """Test generate_response with multiple messages (should use last assistant message)."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "### 4"},
            {"role": "user", "content": "What is 40+2?"},
            {"role": "assistant", "content": "#### 42"},
        ]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert response == "Your response is correct!"
        assert self.interaction._instance_dict[instance_id]["response"] == "#### 42"

    @pytest.mark.asyncio
    async def test_generate_response_no_assistant_message(self):
        """Test generate_response with no assistant messages."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [{"role": "user", "content": "Hello!"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert self.interaction._instance_dict[instance_id]["response"] == ""

    @pytest.mark.asyncio
    async def test_calculate_score_direct_call(self):
        """Test calculate_score method directly."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        # Set a response
        self.interaction._instance_dict[instance_id]["response"] = "#### 42"

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0) as mock_compute:
            score = await self.interaction.calculate_score(instance_id)

            assert score == 1.0
            mock_compute.assert_called_once_with("#### 42", "42", method="strict", format_score=0.0, score=1.0)

    @pytest.mark.asyncio
    async def test_calculate_score_with_kwargs(self):
        """Test calculate_score method with additional kwargs."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        # Set a response
        self.interaction._instance_dict[instance_id]["response"] = "#### 24"

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0) as mock_compute:
            score = await self.interaction.calculate_score(instance_id, extra_param="test")

            assert score == 0.0
            mock_compute.assert_called_once_with("#### 24", "42", method="strict", format_score=0.0, score=1.0)

    @pytest.mark.asyncio
    async def test_finalize_interaction(self):
        """Test finalize_interaction method."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        assert instance_id in self.interaction._instance_dict

        await self.interaction.finalize_interaction(instance_id)

        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_finalize_interaction_with_kwargs(self):
        """Test finalize_interaction method with additional kwargs."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        assert instance_id in self.interaction._instance_dict

        await self.interaction.finalize_interaction(instance_id, extra_param="test")

        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_interaction(self):
        """Test finalize_interaction with non-existent instance_id."""
        instance_id = "nonexistent_instance"

        # This should raise KeyError
        with pytest.raises(KeyError):
            await self.interaction.finalize_interaction(instance_id)

    @pytest.mark.asyncio
    async def test_full_interaction_workflow_correct(self):
        """Test complete interaction workflow with correct answer."""
        ground_truth = "42"

        # Start interaction
        instance_id = await self.interaction.start_interaction(ground_truth=ground_truth)

        # Generate response with correct answer
        messages = [{"role": "assistant", "content": "42"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert reward == 1.0

        # Finalize interaction
        await self.interaction.finalize_interaction(instance_id)
        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_full_interaction_workflow_incorrect(self):
        """Test complete interaction workflow with incorrect answer."""
        ground_truth = "42"

        # Start interaction
        instance_id = await self.interaction.start_interaction(ground_truth=ground_truth)

        # Generate response with incorrect answer
        messages = [{"role": "assistant", "content": "24"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert reward == 0.0

        # Continue with another attempt
        messages.append({"role": "user", "content": response})
        messages.append({"role": "assistant", "content": "42"})

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=1.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is True
        assert reward == 1.0

        # Finalize interaction
        await self.interaction.finalize_interaction(instance_id)
        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_multiple_concurrent_interactions(self):
        """Test multiple concurrent interaction instances."""
        ground_truth_1 = "42"
        ground_truth_2 = "24"

        # Start multiple interactions
        instance_id_1 = await self.interaction.start_interaction(ground_truth=ground_truth_1)
        instance_id_2 = await self.interaction.start_interaction(ground_truth=ground_truth_2)

        assert len(self.interaction._instance_dict) == 2
        assert instance_id_1 in self.interaction._instance_dict
        assert instance_id_2 in self.interaction._instance_dict

        # Test responses for both instances
        messages_1 = [{"role": "assistant", "content": "42"}]
        messages_2 = [{"role": "assistant", "content": "24"}]

        with patch("verl.utils.reward_score.gsm8k.compute_score", side_effect=[1.0, 1.0]):
            should_terminate_1, _, reward_1, _ = await self.interaction.generate_response(instance_id_1, messages_1)
            should_terminate_2, _, reward_2, _ = await self.interaction.generate_response(instance_id_2, messages_2)

        assert should_terminate_1 is True
        assert should_terminate_2 is True
        assert reward_1 == 1.0
        assert reward_2 == 1.0

        # Finalize both interactions
        await self.interaction.finalize_interaction(instance_id_1)
        await self.interaction.finalize_interaction(instance_id_2)

        assert len(self.interaction._instance_dict) == 0

    @pytest.mark.asyncio
    async def test_edge_case_empty_messages(self):
        """Test edge case with empty messages list."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = []

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == ""

    @pytest.mark.asyncio
    async def test_edge_case_message_without_content(self):
        """Test edge case with message without content field."""
        instance_id = "test_instance"
        ground_truth = "42"

        # Setup instance
        await self.interaction.start_interaction(instance_id=instance_id, ground_truth=ground_truth)

        messages = [
            {"role": "assistant"}  # Missing content field
        ]

        with patch("verl.utils.reward_score.gsm8k.compute_score", return_value=0.0):
            should_terminate, response, reward, metadata = await self.interaction.generate_response(
                instance_id, messages
            )

        assert should_terminate is False
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] is None

    def test_inheritance_from_base_interaction(self):
        """Test that Gsm8kInteraction properly inherits from BaseInteraction."""
        from verl.interactions.base import BaseInteraction

        assert isinstance(self.interaction, BaseInteraction)

        # Test that all required methods are implemented
        assert hasattr(self.interaction, "start_interaction")
        assert hasattr(self.interaction, "generate_response")
        assert hasattr(self.interaction, "calculate_score")
        assert hasattr(self.interaction, "finalize_interaction")

        # Test that methods are callable
        assert callable(self.interaction.start_interaction)
        assert callable(self.interaction.generate_response)
        assert callable(self.interaction.calculate_score)
        assert callable(self.interaction.finalize_interaction)

    def test_name_attribute_initialization(self):
        """Test name attribute initialization with different configs."""
        # Test with explicit name in config
        config_with_name = {"name": "custom_gsm8k"}
        interaction_with_name = Gsm8kInteraction(config_with_name)
        assert interaction_with_name.name == "custom_gsm8k"

        # Test with default name when not provided in config
        config_without_name = {}
        interaction_without_name = Gsm8kInteraction(config_without_name)
        assert interaction_without_name.name == "interaction_agent"  # Default from BaseInteraction

        # Test that name is accessible as attribute
        assert hasattr(self.interaction, "name")
        assert self.interaction.name == "gsm8k"
