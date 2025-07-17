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
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from tensordict import TensorDict

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.config.actor import FSDPActorConfig


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing DataParallelPPOActor"""
    
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return MockOutput(logits)


class TestDataParallelPPOActor(unittest.TestCase):
    """Test DataParallelPPOActor compute_log_prob and update_policy methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_distributed_patcher = patch('torch.distributed.get_rank')
        self.mock_get_rank = self.mock_distributed_patcher.start()
        self.mock_get_rank.return_value = 0
        
        self.mock_memory_info_patcher = patch('verl.utils.profiler.performance._get_current_mem_info')
        self.mock_memory_info = self.mock_memory_info_patcher.start()
        self.mock_memory_info.return_value = ("0.00", "0.00", "0.00", "0.00")
        
        self.config = FSDPActorConfig(
            strategy="fsdp",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=0.01,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,  # Disable torch.compile for testing
            ulysses_sequence_parallel_size=1
        )
        
        self.mock_model = MockTransformerModel(vocab_size=1000, hidden_size=64)
        self.mock_optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)
        
        self.actor = DataParallelPPOActor(
            config=self.config,
            actor_module=self.mock_model,
            actor_optimizer=self.mock_optimizer
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.mock_distributed_patcher.stop()
        self.mock_memory_info_patcher.stop()
    
    def _create_test_data_for_compute_log_prob(self):
        """Create test DataProto for compute_log_prob method"""
        batch_size = 2
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, total_length))
        attention_mask = torch.ones(batch_size, total_length)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1)
        responses = input_ids[:, -response_length:]  # Last part is the response
        
        tensor_dict = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses
        }, batch_size=[batch_size])
        
        meta_info = {
            "micro_batch_size": batch_size,
            "temperature": 1.0,
            "use_dynamic_bsz": False
        }
        
        return DataProto(batch=tensor_dict, meta_info=meta_info)
    
    def _create_test_data_for_update_policy(self):
        """Create test DataProto for update_policy method"""
        batch_size = 4  # Must match ppo_mini_batch_size
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, total_length))
        attention_mask = torch.ones(batch_size, total_length)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1)
        responses = input_ids[:, -response_length:]
        response_mask = torch.ones(batch_size, response_length)
        old_log_probs = torch.randn(batch_size, response_length) * 0.1  # Small values
        advantages = torch.randn(batch_size, response_length) * 0.5
        
        tensor_dict = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages
        }, batch_size=[batch_size])
        
        meta_info = {
            "temperature": 1.0
        }
        
        return DataProto(batch=tensor_dict, meta_info=meta_info)
    
    def test_compute_log_prob(self):
        """Test compute_log_prob method"""
        data = self._create_test_data_for_compute_log_prob()
        
        log_probs, entropies = self.actor.compute_log_prob(data, calculate_entropy=True)
        
        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]
        
        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        
        self.assertIsInstance(entropies, torch.Tensor)
        self.assertEqual(entropies.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropies)))
        self.assertTrue(torch.all(entropies >= 0))  # Entropy should be non-negative
    
    def test_compute_log_prob_without_entropy(self):
        """Test compute_log_prob method without entropy calculation"""
        data = self._create_test_data_for_compute_log_prob()
        
        log_probs, entropies = self.actor.compute_log_prob(data, calculate_entropy=False)
        
        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]
        
        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        
        self.assertIsNone(entropies)
    
    def test_update_policy(self):
        """Test update_policy method"""
        data = self._create_test_data_for_update_policy()
        
        metrics = self.actor.update_policy(data)
        
        self.assertIsInstance(metrics, dict)
        
        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac", 
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm"
        ]
        
        for key in expected_metric_keys:
            self.assertIn(key, metrics)
            if isinstance(metrics[key], list):
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics[key]))
            else:
                self.assertIsInstance(metrics[key], (float, int))
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))
    
    def test_dataparallelppoactor_initialization(self):
        """Test DataParallelPPOActor initialization"""
        self.assertIsNotNone(self.actor.actor_module)
        self.assertIsNotNone(self.actor.actor_optimizer)
        self.assertEqual(self.actor.config, self.config)
        
        self.assertEqual(self.actor.config.strategy, "fsdp")
        self.assertEqual(self.actor.config.ppo_mini_batch_size, 4)
        self.assertEqual(self.actor.config.clip_ratio, 0.2)


if __name__ == "__main__":
    unittest.main()
