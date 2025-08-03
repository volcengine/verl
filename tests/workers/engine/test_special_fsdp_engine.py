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

import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config import CheckpointConfig
from verl.workers.config import OptimizerConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine.fsdp import FSDPEngine


class TestFSDPEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
            )
        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(cls.rank)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Set up test fixtures"""

        self.temp_dir = tempfile.mkdtemp()
        self.config = OmegaConf.create(
            {
                "strategy": "fsdp2",
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size_per_gpu": 2,
                "forward_micro_batch_size_per_gpu": 2,
                "ppo_micro_batch_size": None,
                "forward_micro_batch_size": None,
                "ppo_max_token_len_per_gpu": 64,
                "rollout_n": 1,
                "ulysses_sequence_parallel_size": 2,
                "use_dynamic_bsz": False,
                "grad_clip": 1.0,
                "optim": OptimizerConfig(lr=1e-6),
                "model": {
                    "path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "tokenizer_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "fsdp_config": FSDPEngineConfig(
                        param_offload=True,
                        optimizer_offload=True,
                    ),
                    "use_remove_padding": False,
                },
                "checkpoint": CheckpointConfig(),
            }
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_data_proto(self, batch_size: int = 2, seq_len: int = 10, response_len=5) -> DataProto:
        """Create test data"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float)
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

    def test_init_model(self):
        """Test FSDPEngine.init_model() method"""
        engine = FSDPEngine(self.config)
        engine.init_model()

        self.assertIsNotNone(engine.module)
        self.assertIsNotNone(engine.optimizer)
        self.assertIsNotNone(engine.lr_scheduler)
        self.assertIsNotNone(engine.checkpoint_manager)

    def test_infer_batch(self):
        """Test FSDPEngine.infer_batch() method"""
        engine = FSDPEngine(self.config)
        engine.init_model()

        batch_size = 2
        response_len = 5
        data = self._make_data_proto(batch_size=batch_size, response_len=response_len)

        def post_fn(micro_batch, preds):
            response_length = micro_batch["responses"].size(-1)
            values = preds[:, -response_length - 1 : -1]
            return values, {"values": values.clone().detach()}

        with engine.eval_mode():
            data = engine.shard_data(data=data)
            output = engine.infer_batch(data, post_fn=post_fn)
            output = DataProto.from_dict(tensors={"values": output["values"]})
            output = engine.unshard_data(data=output)

        self.assertIn("values", output.batch)
        self.assertEqual(output.batch["values"].shape, (batch_size, response_len, 1))
        self.assertTrue(torch.isfinite(output.batch["values"]).all())

    def test_train_batch(self):
        """Test FSDPEngine.train_batch() method"""
        engine = FSDPEngine(self.config)
        engine.init_model()

        data = self._make_data_proto()

        def loss_fn(micro_batch, preds):
            logits = preds.view(-1, preds.size(-1))
            labels = torch.zeros(preds.shape[0], preds.shape[1], dtype=torch.long, device=logits.device).view(-1)
            loss = F.cross_entropy(logits, labels)
            metrics = {"dummy_loss": loss.detach().clone()}

            return loss, metrics

        with engine.train_mode():
            data = engine.shard_data(data=data)
            engine.optimizer_zero_grad()
            metrics = engine.train_batch(data, loss_fn=loss_fn)
            grad_norm = engine.optimizer_step()
            lr = engine.lr_scheduler_step()
            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = engine.unshard_data(data=output)

        self.assertIn("dummy_loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["dummy_loss"])).all())
        self.assertTrue(torch.isfinite(torch.tensor(grad_norm)).all())
        self.assertTrue(torch.isfinite(torch.tensor(lr)).all())

    def test_shard_unshard_data(self):
        """Test FSDPEngine.shard()/unshard() method"""
        assert self.config.ulysses_sequence_parallel_size == 2
        engine = FSDPEngine(self.config)
        engine.init_model()

        batch_size = 2
        response_len = 5
        data = self._make_data_proto(batch_size=batch_size, response_len=response_len)

        sharded_data = engine.shard_data(data)
        unsharded_data = engine.unshard_data(sharded_data)

        self.assertTrue(sharded_data.batch.batch_size[0], batch_size * self.config.ulysses_sequence_parallel_size)
        self.assertTrue(unsharded_data.batch.batch_size[0], batch_size)

    def test_to_device(self):
        """Test FSDPEngine.to() method"""
        config = OmegaConf.create(self.config)
        config.model.fsdp_config.param_offload = False
        config.model.fsdp_config.optimizer_offload = False
        engine = FSDPEngine(config)
        engine.init_model()

        engine.to("cpu", model=True, optimizer=True)
        param_device = next(engine.module.parameters()).device
        self.assertEqual(param_device.type, "cpu")

        engine.to("cuda", model=True, optimizer=True)
        param_device = next(engine.module.parameters()).device
        self.assertEqual(param_device.type, "cuda")

    def test_save_load_checkpoint(self):
        """Test FSDPEngine.load/save_checkpoint() method"""
        engine = FSDPEngine(self.config)
        engine.init_model()

        engine.save_checkpoint(self.temp_dir, global_step=1)
        engine.load_checkpoint(self.temp_dir)
        rank = torch.distributed.get_rank()
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"model_world_size_2_rank_{rank}.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"optim_world_size_2_rank_{rank}.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"extra_state_world_size_2_rank_{rank}.pt")))


if __name__ == "__main__":
    unittest.main()
