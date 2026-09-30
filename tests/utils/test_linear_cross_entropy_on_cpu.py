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

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "kernel" / "linear_cross_entropy.py"
_SPEC = importlib.util.spec_from_file_location("_linear_cross_entropy", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lce)


def test_triton_backend_preserves_existing_autograd_dispatch(monkeypatch):
    expected = (torch.tensor([1.0]), torch.tensor([2.0]))
    calls = []

    class FakeLinearCrossEntropy:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return expected

    monkeypatch.setattr(lce, "LinearCrossEntropy", FakeLinearCrossEntropy)
    hidden = torch.randn(3, 5)
    weight = torch.randn(7, 5)
    labels = torch.randint(7, (3,))
    group = object()

    output = lce.linear_cross_entropy(
        hidden,
        weight,
        labels,
        0.8,
        "none",
        group,
        impl_backend="triton",
    )

    assert output is expected
    assert calls == [(hidden, weight, labels, 0.8, "none", group)]


def test_liger_tp_uses_fixed_size_padded_chunks(monkeypatch):
    calls = []

    class FakeNativeFunction:
        @staticmethod
        def apply(
            hidden,
            weight,
            labels,
            vocab_start,
            temperature,
            ignore_index,
            tiles_per_reduce,
            return_entropy,
            process_group,
        ):
            call_index = len(calls)
            calls.append(
                {
                    "hidden": hidden,
                    "weight": weight,
                    "labels": labels,
                    "vocab_start": vocab_start,
                    "temperature": temperature,
                    "ignore_index": ignore_index,
                    "tiles_per_reduce": tiles_per_reduce,
                    "return_entropy": return_entropy,
                    "process_group": process_group,
                }
            )
            offset = call_index * 10
            nll = torch.arange(hidden.shape[0], dtype=torch.float32) + offset
            entropy = torch.arange(hidden.shape[0], dtype=torch.float32) + 100 + offset
            return nll, entropy

    monkeypatch.setattr(lce, "_validate_liger_tp_device", lambda: None)
    monkeypatch.setattr(lce, "_require_liger_tp_runtime", lambda: (FakeNativeFunction, object()))
    process_group = object()
    monkeypatch.setattr(lce.dist, "get_rank", lambda group: 2 if group is process_group else -1)

    hidden = torch.arange(35, dtype=torch.float32).reshape(1, 7, 5)
    weight = torch.randn(11, 5)
    labels = torch.arange(7, dtype=torch.int32).reshape(1, 7)

    log_probs, entropy = lce.linear_cross_entropy(
        hidden,
        weight,
        labels,
        0.7,
        "none",
        process_group,
        impl_backend="liger_tp",
        chunk_size=3,
        tiles_per_reduce=2,
    )

    assert len(calls) == 3
    assert [tuple(call["hidden"].shape) for call in calls] == [(3, 5), (3, 5), (3, 5)]
    assert calls[-1]["labels"].tolist() == [6, -100, -100]
    torch.testing.assert_close(calls[-1]["hidden"][1:], torch.zeros(2, 5))
    assert all(call["labels"].dtype == torch.int64 for call in calls)
    assert all(call["vocab_start"] == 22 for call in calls)
    assert all(call["temperature"] == 0.7 for call in calls)
    assert all(call["ignore_index"] == -100 for call in calls)
    assert all(call["tiles_per_reduce"] == 2 for call in calls)
    assert all(call["return_entropy"] is True for call in calls)
    assert all(call["process_group"] is process_group for call in calls)
    torch.testing.assert_close(log_probs, -torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0]))
    torch.testing.assert_close(entropy, torch.tensor([100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 120.0]))


def test_liger_tp_accumulates_gradients_across_padded_chunks(monkeypatch):
    class FakeNativeFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden,
            weight,
            labels,
            vocab_start,
            temperature,
            ignore_index,
            tiles_per_reduce,
            return_entropy,
            process_group,
        ):
            del labels, vocab_start, temperature, ignore_index, tiles_per_reduce, return_entropy, process_group
            ctx.save_for_backward(hidden, weight)
            nll = (hidden * weight[0]).sum(dim=-1)
            entropy = (hidden * weight[1]).sum(dim=-1)
            return nll, entropy

        @staticmethod
        def backward(ctx, grad_nll, grad_entropy):
            hidden, weight = ctx.saved_tensors
            grad_hidden = grad_nll[:, None] * weight[0] + grad_entropy[:, None] * weight[1]
            grad_weight = torch.stack(
                (
                    (grad_nll[:, None] * hidden).sum(dim=0),
                    (grad_entropy[:, None] * hidden).sum(dim=0),
                )
            )
            return grad_hidden, grad_weight, None, None, None, None, None, None, None

    monkeypatch.setattr(lce, "_validate_liger_tp_device", lambda: None)
    monkeypatch.setattr(lce, "_require_liger_tp_runtime", lambda: (FakeNativeFunction, object()))
    monkeypatch.setattr(lce.dist, "get_rank", lambda _group: 0)

    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4).requires_grad_(True)
    weight = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [-1.0, 0.5, 0.25, -0.75],
        ],
        requires_grad=True,
    )
    labels = torch.arange(5)
    grad_log_probs = torch.linspace(-0.5, 0.5, 5)
    grad_entropy = torch.linspace(0.3, -0.2, 5)
    process_group = object()

    log_probs, entropy = lce.linear_cross_entropy(
        hidden,
        weight,
        labels,
        dist_process_group=process_group,
        impl_backend="liger_tp",
        chunk_size=3,
    )
    torch.autograd.backward((log_probs, entropy), (grad_log_probs, grad_entropy))

    expected_hidden_grad = -grad_log_probs[:, None] * weight.detach()[0]
    expected_hidden_grad += grad_entropy[:, None] * weight.detach()[1]
    expected_weight_grad = torch.stack(
        (
            (-grad_log_probs[:, None] * hidden.detach()).sum(dim=0),
            (grad_entropy[:, None] * hidden.detach()).sum(dim=0),
        )
    )
    torch.testing.assert_close(hidden.grad, expected_hidden_grad)
    torch.testing.assert_close(weight.grad, expected_weight_grad)


def test_liger_tp_initialization_is_idempotent_for_same_tp_membership(monkeypatch):
    class FakeNvshmem:
        def __init__(self):
            self.init_calls = []
            self.resolve_calls = []

        def init_from_pg(self, process_group):
            self.init_calls.append(process_group)

        def resolve_team(self, process_group):
            self.resolve_calls.append(process_group)

    nvshmem = FakeNvshmem()
    group = object()
    equivalent_group = object()
    other_group = object()
    memberships = {
        group: (4, 6),
        equivalent_group: (4, 6),
        other_group: (5, 7),
    }

    monkeypatch.setattr(lce, "_LIGER_TP_BOOTSTRAP_GLOBAL_RANKS", None)
    monkeypatch.setattr(lce, "_validate_liger_tp_device", lambda: None)
    monkeypatch.setattr(lce, "_require_liger_tp_runtime", lambda: (object(), nvshmem))
    monkeypatch.setattr(lce.dist, "is_available", lambda: True)
    monkeypatch.setattr(lce.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lce.dist, "get_world_size", lambda process_group: len(memberships[process_group]))
    monkeypatch.setattr(
        lce.dist,
        "get_global_rank",
        lambda process_group, group_rank: memberships[process_group][group_rank],
    )

    lce.initialize_liger_tp_flsce(group)
    lce.initialize_liger_tp_flsce(equivalent_group)

    assert nvshmem.init_calls == [group]
    assert nvshmem.resolve_calls == [group, equivalent_group]

    with pytest.raises(RuntimeError, match="already initialized"):
        lce.initialize_liger_tp_flsce(other_group)


def test_linear_cross_entropy_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported linear cross entropy backend"):
        lce.linear_cross_entropy(
            torch.randn(2, 3),
            torch.randn(5, 3),
            torch.randint(5, (2,)),
            impl_backend="unknown",
        )
