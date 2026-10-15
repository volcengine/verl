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

"""CPU-only tests for keeping a frozen vision tower out of FSDP wrapping.

``actor.freeze_vision_tower`` freezes the vision tower and excludes it from
per-module FSDP wrapping (both FSDP1 and FSDP2), so its params join the root
FSDP unit. These tests verify the wrap-target selection on both FSDP paths
plus the ``get_vl_model_vision_tower`` helper.
"""

from types import SimpleNamespace

import torch.nn as nn
import torch.nn.functional as F

from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets, get_fsdp_wrap_policy
from verl.utils.model import get_vl_model_vision_tower


class _MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)


class _MockVisionBlock(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)


class _MockVisionTower(nn.Module):
    def __init__(self, hidden_size=32, depth=2):
        super().__init__()
        self.patch_embed = nn.Linear(12, hidden_size)
        self.blocks = nn.ModuleList([_MockVisionBlock(hidden_size) for _ in range(depth)])

    def forward(self, pixel_values, grid_thw=None):
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block.fc2(F.gelu(block.fc1(x)))
        return x


class _MockVLM(nn.Module):
    """Minimal vision-language model: vision tower + decoder layers + embed/lm_head."""

    _no_split_modules = ["_MockDecoderLayer", "_MockVisionBlock"]

    def __init__(self, vocab_size=64, hidden_size=32, num_layers=2):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.visual = _MockVisionTower(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_MockDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)


class _VisionOnlyVLM(_MockVLM):
    """A model whose ``_no_split_modules`` lists only the vision block class."""

    _no_split_modules = ["_MockVisionBlock"]


def _wrap_classes_from_policy(policy):
    """Extract the transformer class set from the FSDP1 policy partial chain."""
    or_policy = policy.func
    assert or_policy.__name__ == "_or_policy", "expected _or_policy at the top level"
    transformer_policy = policy.keywords["policies"][-1]
    return transformer_policy.keywords["transformer_layer_cls"]


def test_fsdp1_policy_excludes_frozen_vision_tower():
    model = _MockVLM()
    policy = get_fsdp_wrap_policy(model, exclude_modules=[model.visual])
    classes = _wrap_classes_from_policy(policy)
    assert classes == {_MockDecoderLayer}, f"expected only decoder layers, got {classes}"


def test_fsdp1_policy_without_exclusion_wraps_vision_blocks():
    model = _MockVLM()
    policy = get_fsdp_wrap_policy(model)
    classes = _wrap_classes_from_policy(policy)
    assert classes == {_MockDecoderLayer, _MockVisionBlock}, f"expected both classes, got {classes}"


def test_fsdp1_policy_raises_when_exclusion_removes_all_classes():
    """Excluding the only wrap class must raise, not silently no-op."""
    import pytest

    model = _VisionOnlyVLM()
    assert get_fsdp_wrap_policy(model) is not None, "without exclusion the policy should build"
    with pytest.raises(Exception, match="Could not find any of the transformer layer classes"):
        get_fsdp_wrap_policy(model, exclude_modules=[model.visual])


def test_fsdp2_targets_exclude_frozen_vision_tower():
    model = _MockVLM()
    targets = _select_fsdp2_wrap_targets(
        model, ["_MockDecoderLayer", "_MockVisionBlock"], exclude_modules=[model.visual]
    )
    target_classes = {type(m) for m in targets}
    assert _MockVisionBlock not in target_classes, "vision blocks must not be wrapped"
    assert _MockDecoderLayer in target_classes, "decoder layers must still be wrapped"
    # embed_tokens / lm_head are matched by name and must survive the exclusion.
    assert any(isinstance(m, nn.Embedding) for m in targets), "embed_tokens should remain a wrap target"
    # No module inside the tower may appear.
    tower_ids = {id(sub) for sub in model.visual.modules()}
    assert not (tower_ids & {id(m) for m in targets}), "no tower submodule may be a wrap target"


def test_fsdp2_targets_without_exclusion_include_vision_blocks():
    model = _MockVLM()
    targets = _select_fsdp2_wrap_targets(model, ["_MockDecoderLayer", "_MockVisionBlock"])
    target_classes = {type(m) for m in targets}
    assert _MockVisionBlock in target_classes, "vision blocks should be wrapped without exclusion"
    assert _MockDecoderLayer in target_classes


def test_get_vl_model_vision_tower_layouts():
    tower = _MockVisionTower()

    class _TopLevel(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = tower

    class _Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _TopLevel()

    class _TextOnly(nn.Module):
        pass

    assert get_vl_model_vision_tower(_TopLevel()) is tower
    assert get_vl_model_vision_tower(_Nested()) is tower
    assert get_vl_model_vision_tower(_TextOnly()) is None
