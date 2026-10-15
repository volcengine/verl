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


def test_fsdp1_policy_allows_exclusion_removing_all_classes():
    """Excluding the only wrap class wraps nothing per-module (root unit holds
    everything), matching the FSDP2 path; genuinely missing classes still raise."""
    import pytest

    model = _VisionOnlyVLM()
    assert get_fsdp_wrap_policy(model) is not None, "without exclusion the policy should build"
    assert get_fsdp_wrap_policy(model, exclude_modules=[model.visual]) is None, (
        "exclusion removing all wrap classes must build no per-module policy, not raise"
    )

    class _MissingClassVLM(_MockVLM):
        _no_split_modules = ["_NoSuchLayerClass"]

    with pytest.raises(Exception, match="Could not find any of the transformer layer classes"):
        get_fsdp_wrap_policy(_MissingClassVLM())


def _lora_leaf(where):
    """Attach a trainable LoRA-like linear leaf, as PEFT does."""
    leaf = nn.Linear(32, 32)
    where.lora_A = leaf
    return leaf


def _lambda_fn_from_policy(policy):
    """Extract the LoRA lambda_fn from the FSDP1 policy partial chain."""
    lambda_policy = policy.keywords["policies"][0]
    return lambda_policy.keywords["lambda_fn"]


def test_fsdp1_lora_policy_excludes_leaves_inside_excluded_tower():
    """Trainable LoRA leaves inside the frozen, wrap-excluded tower must not be
    wrapped individually: a rank that skips the tower would never enter those
    units and desynchronize collective order (#3491). They join the root unit."""
    model = _MockVLM()
    model.visual.requires_grad_(False)
    tower_leaf = _lora_leaf(model.visual)
    lang_leaf = _lora_leaf(model.layers[0])

    lambda_fn = _lambda_fn_from_policy(get_fsdp_wrap_policy(model, is_lora=True, exclude_modules=[model.visual]))
    assert lambda_fn(tower_leaf) is False, "tower LoRA leaf must not be individually wrapped"
    assert lambda_fn(lang_leaf) is True, "language LoRA leaf must stay individually wrapped"


def test_fsdp1_lora_policy_without_exclusion_wraps_tower_leaves():
    """Without exclusion (freeze off) the LoRA policy is unchanged."""
    model = _MockVLM()
    tower_leaf = _lora_leaf(model.visual)

    lambda_fn = _lambda_fn_from_policy(get_fsdp_wrap_policy(model, is_lora=True))
    assert lambda_fn(tower_leaf) is True, "no exclusion: LoRA leaves are wrapped as before"


def test_mark_frozen_vision_tower_for_skip_gates_on_trainable_params():
    """The engine marks only towers with no trainable params inside: with the
    tower wrap-excluded into the root FSDP unit, a trainable param inside would
    get grads on image ranks only on mixed-modality batches, which FSDP1's
    root-unit gradient reduce does not support (verified to hang)."""
    from verl.utils.fsdp_utils import mark_frozen_vision_tower_for_skip

    tower = _MockVisionTower()
    tower.requires_grad_(False)
    assert mark_frozen_vision_tower_for_skip(tower) is True
    assert getattr(tower, "_verl_frozen_vision_tower", False) is True

    tower_with_adapter = _MockVisionTower()
    tower_with_adapter.requires_grad_(False)
    _lora_leaf(tower_with_adapter)  # trainable leaf, as PEFT attaches
    assert mark_frozen_vision_tower_for_skip(tower_with_adapter) is False
    assert not hasattr(tower_with_adapter, "_verl_frozen_vision_tower")


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
