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

import tempfile
import unittest

import torch
import torch.distributed
from tensordict import TensorDict
from transformers import AutoConfig

from verl import DataProto
from verl.workers.config import FSDPCriticConfig, OptimizerConfig
from verl.workers.config.critic import FSDPCriticModelCfg
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.fsdp_workers import CriticWorker


class TestCriticWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()

        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        config.save_pretrained(self.temp_dir)

        self.config = FSDPCriticConfig(
            strategy="fsdp2",
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
                path="Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct",
                fsdp_config=FSDPEngineConfig(fsdp_size=-1),
                use_remove_padding=False,
            ),
        )
        assert self.world_size <= 4 // 2

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data_for_compute_values(self, batch_size=2, seq_len=10, response_len=5):
        """Create test data for compute_values method"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        responses = torch.randint(0, 1000, (batch_size, response_len), dtype=torch.long)
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            batch_size=[batch_size],
        )

        data = DataProto(
            batch=batch, meta_info={"micro_batch_size": 2, "max_token_len": seq_len, "use_dynamic_bsz": False}
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

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "values": values,
                "returns": returns,
            },
            batch_size=[batch_size],
        )

        data = DataProto(
            batch=batch,
            meta_info={"global_token_num": [response_len] * batch_size, "batch_seqlens": [response_len] * batch_size},
        )

        return data

    def test_init_model(self):
        """Test CriticWorker.init_model() method"""
        worker = CriticWorker(self.config)
        worker.init_model()

        self.assertIsNotNone(worker.critic_module)
        self.assertIsNotNone(worker.critic_optimizer)
        self.assertIsNotNone(worker.critic)
        self.assertIsNotNone(worker.checkpoint_manager)

    def test_compute_values(self):
        """Test CriticWorker.compute_values() method"""
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

    def test_update_critic(self):
        """Test CriticWorker.update_critic() method"""
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
            if isinstance(value, list | tuple):
                for v in value:
                    self.assertTrue(torch.isfinite(torch.tensor(v)).all())
            else:
                self.assertTrue(torch.isfinite(torch.tensor(value)).all())


if __name__ == "__main__":
    unittest.main()
