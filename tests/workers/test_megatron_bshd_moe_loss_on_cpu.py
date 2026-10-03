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

"""Exercise the real per-token loss callback without importing CUDA-only Megatron.

Only the callback and token-count method are compiled from the engine module;
no source-text assertions or copies of their implementation are used. Model
forward/backward and Megatron's gradient finalization are outside this test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.megatron.moe_loss import validate_bshd_moe_router_loss_compatibility


@pytest.fixture
def engine() -> Any:
    path = Path(__file__).parents[2] / "verl/workers/engine/megatron/transformer_impl.py"
    tree = ast.parse(path.read_text())
    methods = [
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name in ("postprocess_micro_batch_func", "_routed_num_tokens")
    ]
    namespace = {
        "torch": torch,
        "TensorDict": TensorDict,
        "tu": tu,
        "detach_tree": lambda output: output,
        "validate_bshd_moe_router_loss_compatibility": validate_bshd_moe_router_loss_compatibility,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    cls = type("LossCallbackEngine", (), {method.name: namespace[method.name] for method in methods})
    result = cls()
    result.engine_config = SimpleNamespace(
        use_remove_padding=False, dynamic_context_parallel=False, context_parallel_size=2
    )
    result.prepare_model_outputs = lambda output, data: output
    result.get_data_parallel_group = lambda: None
    return result


def _run_loss(engine: Any, forward_only: bool = False, with_loss: bool = True) -> tuple:
    data = TensorDict(
        {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "attention_mask": torch.tensor([[1] * 8, [1] * 4 + [0] * 4]),
        },
        batch_size=[],
    )
    tu.assign_non_tensor(data, num_micro_batch=3, routed_num_tokens=24, dp_size=2)
    value = torch.tensor(2.0, requires_grad=True)
    engine.loss_value = value
    loss_fn = (lambda **kwargs: (value, {})) if with_loss else None
    return engine.postprocess_micro_batch_func({}, data, forward_only, loss_fn)


@pytest.mark.parametrize(
    ("routing_type", "aux_coeff"),
    [
        ("none", 0.1),
        ("sinkhorn", 0.0),
        ("aux_loss", 0.0),
        ("seq_aux_loss", 0.0),
        ("global_aux_loss", 0.0),
        (["aux_loss", "global_aux_loss"], [0.0, 0.0]),
        (["none", "aux_loss"], [0.1, 0.0]),
        (["aux_loss", "global_aux_loss"], 0.0),
    ],
)
@pytest.mark.parametrize("z_coeff", [None, 0.0])
def test_bshd_inactive_router_loss_keeps_per_token_scaling(
    engine: Any, routing_type: str | list[str], aux_coeff: float | list[float], z_coeff: float | None
) -> None:
    engine.tf_config = SimpleNamespace(
        calculate_per_token_loss=True,
        moe_router_load_balancing_type=routing_type,
        moe_aux_loss_coeff=aux_coeff,
        moe_z_loss_coeff=z_coeff,
    )
    loss_sum, num_tokens, output = _run_loss(engine)
    torch.testing.assert_close(loss_sum, torch.tensor(24.0))
    assert num_tokens.item() == 6
    assert num_tokens.dtype == torch.int
    assert output["loss"] == 2.0
    loss_sum.backward()
    torch.testing.assert_close(engine.loss_value.grad, torch.tensor(12.0))


@pytest.mark.parametrize(
    ("routing_type", "aux_coeff", "z_coeff"),
    [
        ("aux_loss", 0.01, None),
        ("seq_aux_loss", 0.01, None),
        ("global_aux_loss", 0.01, None),
        (["none", "global_aux_loss"], [0.0, 0.01], None),
        (["seq_aux_loss", "aux_loss"], [0.01, 0.0], None),
        (["aux_loss", "global_aux_loss"], 0.01, None),
        ("none", 0.0, 0.01),
    ],
)
def test_bshd_active_router_loss_is_rejected(
    engine: Any, routing_type: str | list[str], aux_coeff: float | list[float], z_coeff: float | None
) -> None:
    engine.tf_config = SimpleNamespace(
        calculate_per_token_loss=True,
        moe_router_load_balancing_type=routing_type,
        moe_aux_loss_coeff=aux_coeff,
        moe_z_loss_coeff=z_coeff,
    )
    with pytest.raises(ValueError, match="aux/z loss"):
        _run_loss(engine)
    engine.engine_config.use_remove_padding = True
    assert len(_run_loss(engine)) == 3


def test_dense_model_without_router_fields(engine: Any) -> None:
    engine.tf_config = SimpleNamespace(calculate_per_token_loss=True)
    assert len(_run_loss(engine)) == 3


def test_non_per_token_and_forward_only_paths_are_unchanged(engine: Any) -> None:
    engine.tf_config = SimpleNamespace(
        calculate_per_token_loss=False, moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=0.01
    )
    loss, _ = _run_loss(engine)
    torch.testing.assert_close(loss, torch.tensor(6.0))
    engine.tf_config.calculate_per_token_loss = True
    assert len(_run_loss(engine, forward_only=True, with_loss=False)) == 2
