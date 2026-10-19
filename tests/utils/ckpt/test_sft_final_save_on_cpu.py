# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Exercise both trainer loops without initializing distributed GPU engines."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _fit_method(mode):
    filename = "sft_trainer.py" if mode == "spmd" else "sft_trainer_ray.py"
    path = Path(__file__).resolve().parents[3] / "verl" / "trainer" / filename
    trainer = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    fit = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    module = ast.Module(body=[fit], type_ignores=[])
    scope = {
        "torch": torch,
        "tu": SimpleNamespace(
            get_tensordict=lambda tensor_dict, **kwargs: tensor_dict,
            assign_non_tensor=lambda *args, **kwargs: None,
            get=lambda value, key: value[key],
        ),
        "Tracking": MagicMock(),
        "OmegaConf": SimpleNamespace(to_container=lambda *args, **kwargs: {}),
        "logger": None,
        "log_with_rank": MagicMock(),
        "NonTensorData": lambda value: value,
        "tqdm": lambda iterable, **kwargs: iterable,
        "aggressive_empty_cache": MagicMock(),
        "log_gpu_memory_usage": MagicMock(),
    }
    exec(compile(module, str(path), "exec"), scope)
    return scope["fit"]


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize(
    "epochs,target,resume,save_freq,expected_saves",
    [
        (1, 3, 0, 1, [(1, False), (2, True)]),  # Epoch limit first, on a periodic save.
        (1, 3, 0, -1, [(2, True)]),  # Final save even when periodic saving is disabled.
        (2, 5, 0, 3, [(3, False), (4, True)]),  # Final step outside the periodic cadence.
        (2, 5, 3, 1, [(4, True)]),  # Mid-epoch resume uses the global step.
        (2, 3, 0, 1, [(1, False), (2, False), (3, True)]),  # Step limit first.
    ],
)
def test_sft_final_save_drains_at_either_training_limit(mode, epochs, target, resume, save_freq, expected_saves):
    trainer = SimpleNamespace(
        engine=SimpleNamespace(is_mp_src_rank_with_outputs=lambda: False, get_data_parallel_rank=lambda: 0),
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                total_epochs=epochs, project_name="test", experiment_name="final-save", logger=[], balance_batch=False
            ),
            model=SimpleNamespace(use_remove_padding=True),
            data=SimpleNamespace(
                use_dynamic_bsz=False, max_token_len_per_gpu=128, micro_batch_size_per_gpu=1, pad_mode="no_padding"
            ),
        ),
        model_config=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        checkpoint_config=SimpleNamespace(async_save=True),
        resume_global_step=resume,
        steps_per_epoch=2,
        total_training_steps=target,
        global_batch_size=4,
        rank=0,
        device_name="cpu",
        start_profile_step=-1,
        end_profile_step=-1,
        test_freq=-1,
        save_freq=save_freq,
        val_dataloader=None,
        train_sampler=MagicMock(),
        train_dataloader=[{}] * (2 - resume % 2),
    )
    trainer._get_batch_seqlens = lambda **kwargs: [1] if mode == "spmd" else torch.tensor([1])
    trained = []
    saves = []
    pending = []

    def train_batch(*args, **kwargs):
        trained.append(True)
        output = {"metrics": {"loss": 1.0, "grad_norm": 1.0, "lr": 0.1, "mfu": 0.0}}
        return output if mode == "spmd" else SimpleNamespace(get=lambda: output)

    def save_checkpoint(step, blocking=False):
        saves.append((step, blocking))
        pending.append(step)
        if blocking:
            pending.clear()

    trainer.training_client = SimpleNamespace(train_batch=train_batch, step_profile=lambda: None)
    trainer.ckpt_handler = SimpleNamespace(save_checkpoint=save_checkpoint)
    _fit_method(mode)(trainer)

    assert saves == expected_saves
    assert not pending
    assert len(trained) == min(target, epochs * 2) - resume
    assert trainer.total_training_steps == target
