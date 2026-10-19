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
"""Activation offload must bind every differentiable layer output to the group commit node."""

import copy

import torch
from torch import nn

from verl.utils.activation_offload import (
    ActivationHandler,
    CpuOffloadHookWithOffloadHandler,
    FSDPParameterFilter,
    OffloadHandler,
    group_prefetch_offload_commit,
)


class _RecordingOffloadHandler(OffloadHandler):
    """Enforces the ordering the real handler depends on.

    The real handler replaces a group's saved activations with (key, shape)
    descriptors when it offloads them and only restores tensors in
    ``on_group_commit_backward``. A ``tensor_pop`` before that point is the
    ``assert not isinstance(tensor, tuple)`` failure from the issue, so this
    handler raises in the same situation.
    """

    def __init__(self):
        super().__init__()
        self.saved = {}
        self.events = []
        self.reloaded = False

    def tensor_push(self, tensor, **kwargs):
        tag = len(self.saved)
        self.saved[tag] = tensor
        return tag

    def tensor_pop(self, tensor_tag, **kwargs):
        self.events.append("pop")
        if not self.reloaded:
            raise AssertionError("saved activation popped before the group was reloaded")
        return self.saved.pop(tensor_tag)

    def on_group_commit_forward(self):
        self.events.append("commit_forward")

    def on_group_commit_backward(self):
        self.events.append("commit_backward")
        self.reloaded = True


class _TwoOutputLayer(nn.Module):
    """Returns hidden states plus a recurrent state and a mask, like a gated delta net block."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.state_proj = nn.Linear(8, 4)

    def forward(self, x):
        hidden = torch.tanh(self.proj(x))
        state = self.state_proj(hidden)
        mask = hidden > 0
        return hidden, state, mask, None


def _wrap_with_offload(layer: nn.Module, handler: OffloadHandler) -> None:
    def sync_func(*tensors):
        return group_prefetch_offload_commit(handler, *tensors)

    activation_handler = ActivationHandler(
        CpuOffloadHookWithOffloadHandler(handler), sync_func, FSDPParameterFilter(), enable_ckpt=False
    )
    activation_handler.wrap_module_forward_method(layer)


def test_backward_through_secondary_output_reloads_group_first():
    torch.manual_seed(0)
    handler = _RecordingOffloadHandler()
    layer = _TwoOutputLayer()
    _wrap_with_offload(layer, handler)
    x = torch.randn(2, 8, requires_grad=True)

    _, state, _, _ = layer(x)
    # The gradient enters only through the recurrent state, never through hidden states.
    state.sum().backward()

    assert handler.events.count("commit_forward") == 1
    assert handler.events.count("commit_backward") == 1
    assert handler.events.index("commit_backward") < handler.events.index("pop")
    assert x.grad is not None


def test_wrapped_layer_preserves_outputs_and_structure():
    torch.manual_seed(0)
    layer = _TwoOutputLayer()
    reference = copy.deepcopy(layer)
    _wrap_with_offload(layer, _RecordingOffloadHandler())
    x = torch.randn(2, 8, requires_grad=True)

    outputs = layer(x)
    expected = reference(x)

    assert len(outputs) == len(expected)
    torch.testing.assert_close(outputs[0], expected[0])
    torch.testing.assert_close(outputs[1], expected[1])
    assert outputs[2].dtype == torch.bool
    assert torch.equal(outputs[2], expected[2])
    assert outputs[3] is None
    assert outputs[0].requires_grad and outputs[1].requires_grad
