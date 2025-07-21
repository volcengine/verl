#!/usr/bin/env python3
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
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.distributed
from tensordict import TensorDict
from transformers import AutoConfig
from verl import DataProto
from verl.workers.fsdp_workers import CriticWorker
from verl.workers.config import FSDPCriticConfig, OptimizerConfig
from verl.workers.config.critic import FSDPCriticModelCfg
from verl.workers.config.engine import FSDPEngineConfig


class MockValueHeadModel(nn.Module):
    """Mock model that mimics a value head model for critic testing"""
    
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True), 
            num_layers=2
        )
        self.v_head = nn.Linear(hidden_size, 1)  # Value head for critic
        
        for param in self.parameters():
            param.requires_grad_(True)
        
        class MockConfig:
            def __init__(self):
                self.num_attention_heads = 4
                self.num_key_value_heads = 4
                self.text_config = self
                self.model_type = "gpt2"
            
            def save_pretrained(self, path):
                """Mock method for checkpoint compatibility"""
                pass
                
        self.config = MockConfig()
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Mock method for gradient checkpointing"""
        pass
    
    def to(self, device):
        """Mock method for device placement"""
        super().to(device)
        return self
    
    def can_generate(self):
        """Mock method for checkpoint compatibility"""
        return True
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.v_head(x)
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
            
            def __getitem__(self, index):
                if index == 2:
                    values = torch.randn(self.logits.shape[0], self.logits.shape[1], 1, requires_grad=True)
                    return values
                elif index == 0:
                    return self.logits
                elif index == 1:
                    return None
                else:
                    raise IndexError(f"Index {index} out of range")
        
        return MockOutput(logits)


class TestCriticWorker(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://"
            )
        
        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()
        
        if torch.cuda.is_available():
            torch.cuda.set_device(cls.rank)
            cls.device = torch.device(f"cuda:{cls.rank}")
        else:
            cls.device = torch.device("cpu")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_memory_info_patcher = patch("verl.utils.profiler.performance._get_current_mem_info")
        self.mock_memory_info = self.mock_memory_info_patcher.start()
        self.mock_memory_info.return_value = ("0.00", "0.00", "0.00", "0.00")
        
        from unittest.mock import MagicMock
        mock_device = MagicMock()
        mock_device.get_device_name.return_value = "CPU"
        self.mock_get_torch_device_patcher = patch('verl.utils.flops_counter.get_torch_device', return_value=mock_device)
        self.mock_get_torch_device_patcher.start()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.temp_dir = tempfile.mkdtemp()
        
        config = AutoConfig.from_pretrained("microsoft/DialoGPT-medium")
        config.save_pretrained(self.temp_dir)
        
        self.config = FSDPCriticConfig(
            strategy="fsdp",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            forward_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            cliprange_value=0.5,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            ulysses_sequence_parallel_size=1,
            rollout_n=1,
            optim=OptimizerConfig(lr=1e-6),
            model=FSDPCriticModelCfg(
                path=self.temp_dir,
                tokenizer_path=self.temp_dir,
                fsdp_config=FSDPEngineConfig(fsdp_size=-1),
                use_remove_padding=False
            )
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.mock_memory_info_patcher.stop()
        self.mock_get_torch_device_patcher.stop()
    
    def _create_test_data_for_compute_values(self, batch_size=2, seq_len=10, response_len=5):
        """Create test data for compute_values method"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        responses = torch.randint(0, 1000, (batch_size, response_len), dtype=torch.long)
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float)
        
        batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        }, batch_size=[batch_size])
        
        data = DataProto(
            batch=batch,
            meta_info={
                "micro_batch_size": 2,
                "max_token_len": seq_len,
                "use_dynamic_bsz": False
            }
        )
        
        return data
    
    def _create_test_data_for_update_critic(self, batch_size=2, seq_len=10, response_len=5):
        """Create test data for update_critic method"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        responses = torch.randint(0, 1000, (batch_size, response_len), dtype=torch.long)
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float)
        values = torch.randn(batch_size, response_len, dtype=torch.float)
        returns = torch.randn(batch_size, response_len, dtype=torch.float)
        
        batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "values": values,
            "returns": returns,
        }, batch_size=[batch_size])
        
        data = DataProto(
            batch=batch,
            meta_info={
                "global_token_num": [response_len] * batch_size,
                "batch_seqlens": [response_len] * batch_size
            }
        )
        
        return data
    
    @patch('verl.workers.fsdp_workers.FSDP')
    @patch('verl.utils.model.load_valuehead_model')
    @patch('verl.workers.fsdp_workers.copy_to_local')
    @patch('verl.workers.fsdp_workers.hf_tokenizer')
    @patch('verl.workers.fsdp_workers.hf_processor')
    def test_init_model(self, mock_processor, mock_tokenizer, mock_copy_to_local, mock_load_model, mock_fsdp):
        """Test CriticWorker.init_model() method"""
        mock_copy_to_local.return_value = self.temp_dir
        mock_tokenizer.return_value = type('MockTokenizer', (), {
            'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
            'save_pretrained': lambda self, path: None
        })()
        mock_processor.return_value = None
        mock_load_model.return_value = MockValueHeadModel().to(self.device)
        mock_fsdp.side_effect = lambda module, **kwargs: module
        
        worker = CriticWorker(self.config)
        worker.init_model()
        
        self.assertIsNotNone(worker.critic_module)
        self.assertIsNotNone(worker.critic_optimizer)
        self.assertIsNotNone(worker.critic)
        self.assertIsNotNone(worker.checkpoint_manager)
    
    @patch('verl.workers.fsdp_workers.FSDP')
    @patch('verl.utils.model.load_valuehead_model')
    @patch('verl.workers.fsdp_workers.copy_to_local')
    @patch('verl.workers.fsdp_workers.hf_tokenizer')
    @patch('verl.workers.fsdp_workers.hf_processor')
    def test_compute_values(self, mock_processor, mock_tokenizer, mock_copy_to_local, mock_load_model, mock_fsdp):
        """Test CriticWorker.compute_values() method"""
        mock_copy_to_local.return_value = self.temp_dir
        mock_tokenizer.return_value = type('MockTokenizer', (), {
            'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
            'save_pretrained': lambda self, path: None
        })()
        mock_processor.return_value = None
        mock_load_model.return_value = MockValueHeadModel().to(self.device)
        mock_fsdp.side_effect = lambda module, **kwargs: module
        
        worker = CriticWorker(self.config)
        worker.init_model()
        
        data = self._create_test_data_for_compute_values()
        
        result = worker.compute_values(data)
        
        self.assertIsInstance(result, DataProto)
        self.assertIn("values", result.batch)
        values = result.batch["values"]
        
        batch_size, response_len = 2, 5
        self.assertEqual(values.shape, (batch_size, response_len))
        
        self.assertTrue(torch.isfinite(values).all())
    
    @patch('verl.workers.fsdp_workers.FSDP')
    @patch('verl.utils.model.load_valuehead_model')
    @patch('verl.workers.fsdp_workers.copy_to_local')
    @patch('verl.workers.fsdp_workers.hf_tokenizer')
    @patch('verl.workers.fsdp_workers.hf_processor')
    def test_update_critic(self, mock_processor, mock_tokenizer, mock_copy_to_local, mock_load_model, mock_fsdp):
        """Test CriticWorker.update_critic() method"""
        mock_copy_to_local.return_value = self.temp_dir
        mock_tokenizer.return_value = type('MockTokenizer', (), {
            'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
            'save_pretrained': lambda self, path: None
        })()
        mock_processor.return_value = None
        mock_load_model.return_value = MockValueHeadModel().to(self.device)
        mock_fsdp.side_effect = lambda module, **kwargs: module
        
        worker = CriticWorker(self.config)
        worker.init_model()
        
        data = self._create_test_data_for_update_critic()
        
        result = worker.update_critic(data)
        
        self.assertIsInstance(result, DataProto)
        self.assertIn("metrics", result.meta_info)
        metrics = result.meta_info["metrics"]
        
        expected_keys = ["critic/vf_loss", "critic/vf_clipfrac", "critic/vpred_mean", "critic/grad_norm"]
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                for v in value:
                    self.assertTrue(torch.isfinite(torch.tensor(v)).all())
            else:
                self.assertTrue(torch.isfinite(torch.tensor(value)).all())
    
    @patch('verl.workers.fsdp_workers.FSDP')
    @patch('verl.utils.model.load_valuehead_model')
    @patch('verl.workers.fsdp_workers.copy_to_local')
    @patch('verl.workers.fsdp_workers.hf_tokenizer')
    @patch('verl.workers.fsdp_workers.hf_processor')
    def test_save_checkpoint(self, mock_processor, mock_tokenizer, mock_copy_to_local, mock_load_model, mock_fsdp):
        """Test CriticWorker.save_checkpoint() method"""
        mock_copy_to_local.return_value = self.temp_dir
        mock_tokenizer.return_value = type('MockTokenizer', (), {
            'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
            'save_pretrained': lambda self, path: None
        })()
        mock_processor.return_value = None
        mock_load_model.return_value = MockValueHeadModel().to(self.device)
        mock_fsdp.side_effect = lambda module, **kwargs: module
        
        worker = CriticWorker(self.config)
        worker.init_model()
        
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            worker.save_checkpoint(local_path=checkpoint_dir, global_step=1)
            checkpoint_saved = True
        except Exception as e:
            checkpoint_saved = False
            print(f"Save checkpoint failed: {e}")
        
        self.assertTrue(checkpoint_saved, "Checkpoint save should succeed")
    
    @patch('verl.workers.fsdp_workers.FSDP')
    @patch('verl.utils.model.load_valuehead_model')
    @patch('verl.workers.fsdp_workers.copy_to_local')
    @patch('verl.workers.fsdp_workers.hf_tokenizer')
    @patch('verl.workers.fsdp_workers.hf_processor')
    def test_load_checkpoint(self, mock_processor, mock_tokenizer, mock_copy_to_local, mock_load_model, mock_fsdp):
        """Test CriticWorker.load_checkpoint() method"""
        mock_copy_to_local.return_value = self.temp_dir
        mock_tokenizer.return_value = type('MockTokenizer', (), {
            'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
            'save_pretrained': lambda self, path: None
        })()
        mock_processor.return_value = None
        mock_load_model.return_value = MockValueHeadModel().to(self.device)
        mock_fsdp.side_effect = lambda module, **kwargs: module
        
        worker = CriticWorker(self.config)
        worker.init_model()
        
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            worker.save_checkpoint(local_path=checkpoint_dir, global_step=1)
            
            worker.load_checkpoint(local_path=checkpoint_dir)
            checkpoint_loaded = True
        except Exception as e:
            checkpoint_loaded = False
            print(f"Load checkpoint failed: {e}")
        
        self.assertTrue(checkpoint_loaded, "Checkpoint load should succeed")


if __name__ == "__main__":
    unittest.main()
