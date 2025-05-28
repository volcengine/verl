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
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import asyncio

class TestGenerativeRewardManager(unittest.TestCase):
    @patch('yaml.safe_load')
    @patch('builtins.open')
    def setUp(self, mock_open, mock_yaml_load):
        # Import the necessary classes
        from transformers import PreTrainedTokenizer
        from verl import DataProto
        from verl.workers.reward_manager import GenerativeRewardManager
        # Mock the yaml config loading
        mock_yaml_load.return_value = {
            "scoring_prompt": "Score the response based on the reference from range 1 - 5. Embrace your score with \\bold{}. Only the \\bold score is needed. {response}",
            "timeout": 30,
            "api_key": "sk-1232434",
            "api_base": "http://128.0.0.1:8000/v1",
            "max_retries": 3,
            "tokenizer": None,
            "apply_chat_template": False,
            "default_score": 0.0,
        }
        
        # Create mock tokenizer
        self.tokenizer = MagicMock(spec=PreTrainedTokenizer)
        self.tokenizer.batch_decode.return_value = ["Answer 1", "Answer 2"]
        
        # Create the reward manager instance
        self.reward_manager = GenerativeRewardManager(
            tokenizer=self.tokenizer,
            num_examine=2,
            reward_fn_key="data_source"
        )
        
        # Set up sample data for testing
        self.data = MagicMock()
        
        # Set up batch data
        self.data.batch = {
            "prompts": torch.tensor([[101, 102], [103, 104]]),
            "responses": torch.tensor([[201, 202], [203, 204]])
        }
        
        # Create individual data items for iteration
        item1 = MagicMock()
        item1.non_tensor_batch = {"reward_model": {"ground_truth": "Expected 1"}}
        
        item2 = MagicMock()
        item2.non_tensor_batch = {"reward_model": {"ground_truth": "Expected 2"}}
        
        # Properly set up iteration for the mock
        self.data.__iter__ = MagicMock(return_value=iter([item1, item2]))
        
        # Set up non_tensor_batch
        self.data.non_tensor_batch = {
            "data_source": ["source1", "source2"],
            "reward_model": {"ground_truth": ["Expected 1", "Expected 2"]},
            "extra_info": None
        }
    
    @patch('asyncio.run')
    def test_verify(self, mock_asyncio_run):
        # Mock the async process_data_async function to return scores
        mock_asyncio_run.return_value = [0.75, 0.85]
        
        # Call verify method
        scores = self.reward_manager.verify(self.data)
        # Check results
        self.assertEqual(scores, [0.75, 0.85])
        self.assertIn("acc", self.data.batch)
        # Check that acc tensor was set with the correct values
        torch.testing.assert_close(
            self.data.batch["acc"], 
            torch.tensor([0.75, 0.85], dtype=torch.float32)
        )
        
        # Verify that tokenizer was called correctly
        self.tokenizer.batch_decode.assert_called_once_with(
            self.data.batch["responses"], 
            skip_special_tokens=True
        )
    
    @patch('asyncio.run')
    def test_verify_handles_timeout(self, mock_asyncio_run):
        # Simulate timeout error in async call
        mock_asyncio_run.side_effect = asyncio.TimeoutError("Timeout occurred")
        
        # Call verify method
        scores = self.reward_manager.verify(self.data)
        
        # Should gracefully handle the timeout and return zeros
        self.assertEqual(scores, [0.0, 0.0])
        self.assertIn("acc", self.data.batch)
        torch.testing.assert_close(
            self.data.batch["acc"], 
            torch.tensor([0.0, 0.0], dtype=torch.float32)
        )
    
    @patch('asyncio.run')
    def test_call_method(self, mock_asyncio_run):
        # Setup for the __call__ test
        data = MagicMock()
        
        # Case 1: When rm_scores already exists
        data.batch = {"rm_scores": torch.tensor([0.9, 0.8])}
        result = self.reward_manager(data)
        self.assertEqual(result.tolist(), data.batch["rm_scores"].tolist())
        
        # Case 2: When we need to compute rewards
        # Mock the verify method to control its output
        self.reward_manager.verify = MagicMock(return_value=[0.8, 0.9])
        
        data.batch = {
            "prompts": torch.tensor([[101, 102], [103, 104]]),
            "responses": torch.tensor([[201, 202], [203, 204]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
        }
        
        # Create valid_response_length mock
        data.batch["attention_mask"].__getitem__ = MagicMock(
            return_value=torch.tensor([[1, 1], [1, 1]])
        )
        data.batch["attention_mask"].__getitem__.return_value.sum = MagicMock(
            return_value=torch.tensor([2, 2])
        )
        
        data.non_tensor_batch = {"data_source": ["source1", "source2"]}
        
        # Call the method
        with patch.object(torch, 'zeros_like', return_value=torch.zeros((2, 2))):
            result = self.reward_manager(data)
            
            # Verify reward_manager.verify was called
            self.reward_manager.verify.assert_called_once_with(data)
            
            # Test with return_dict=True
            result_dict = self.reward_manager(data, return_dict=True)
            self.assertIsInstance(result_dict, dict)
            self.assertIn("reward_tensor", result_dict)