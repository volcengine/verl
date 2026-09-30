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

from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore import model_forward_fused
from verl.utils.kernel import linear_cross_entropy
from verl.workers.engine.megatron import transformer_impl


@pytest.mark.parametrize(
    ("configured_backend", "expected_backend"),
    [
        ("torch", "triton"),
        ("triton", "triton"),
        ("liger", "liger_tp"),
        ("liger_tp", "liger_tp"),
    ],
)
def test_resolve_megatron_fused_kernel_backend_aliases(configured_backend, expected_backend):
    model_config = SimpleNamespace(
        fused_kernel_options={
            "impl_backend": configured_backend,
            "chunk_size": 256,
            "tiles_per_reduce": 2,
        }
    )

    assert transformer_impl._resolve_megatron_fused_kernel_options(model_config) == (
        expected_backend,
        256,
        2,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"impl_backend": "unknown"},
        {"impl_backend": "liger_tp", "chunk_size": 0},
        {"impl_backend": "liger_tp", "tiles_per_reduce": 3},
    ],
)
def test_resolve_megatron_fused_kernel_options_rejects_invalid_values(options):
    with pytest.raises((TypeError, ValueError)):
        transformer_impl._resolve_megatron_fused_kernel_options(SimpleNamespace(fused_kernel_options=options))


def test_megatron_engine_initializes_and_patches_liger_tp(monkeypatch):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(use_fused_kernels=True, use_remove_padding=True)
    engine.model_config = SimpleNamespace(
        fused_kernel_options={
            "impl_backend": "liger_tp",
            "chunk_size": 1024,
            "tiles_per_reduce": 4,
        },
        mtp=SimpleNamespace(enable=False),
    )
    engine.is_value_model = False
    engine.param_dtype = torch.bfloat16
    engine.module = [
        SimpleNamespace(post_process=True),
        SimpleNamespace(post_process=True),
    ]
    process_group = object()
    init_calls = []
    patch_calls = []

    monkeypatch.setattr(
        transformer_impl.mpu,
        "get_tensor_model_parallel_group",
        lambda: process_group,
    )
    monkeypatch.setattr(
        linear_cross_entropy,
        "initialize_liger_tp_flsce",
        lambda group: init_calls.append(group),
    )
    monkeypatch.setattr(
        model_forward_fused,
        "patch_fused_forward",
        lambda model, **kwargs: patch_calls.append((model, kwargs)),
    )

    engine._maybe_enable_fused_kernels()

    assert init_calls == [process_group]
    assert patch_calls == [
        (
            engine.module[0],
            {
                "impl_backend": "liger_tp",
                "chunk_size": 1024,
                "tiles_per_reduce": 4,
            },
        ),
        (
            engine.module[1],
            {
                "impl_backend": "liger_tp",
                "chunk_size": 1024,
                "tiles_per_reduce": 4,
            },
        ),
    ]


def test_megatron_engine_skips_liger_tp_initialization_without_output_head(monkeypatch):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(use_fused_kernels=True, use_remove_padding=True)
    engine.model_config = SimpleNamespace(
        fused_kernel_options={"impl_backend": "liger_tp"},
        mtp=SimpleNamespace(enable=False),
    )
    engine.is_value_model = False
    engine.param_dtype = torch.bfloat16
    engine.module = [SimpleNamespace(post_process=False)]
    init_calls = []
    patch_calls = []

    monkeypatch.setattr(
        linear_cross_entropy,
        "initialize_liger_tp_flsce",
        lambda group: init_calls.append(group),
    )
    monkeypatch.setattr(
        model_forward_fused,
        "patch_fused_forward",
        lambda model, **kwargs: patch_calls.append((model, kwargs)),
    )

    engine._maybe_enable_fused_kernels()

    assert init_calls == []
    assert patch_calls == [
        (
            engine.module[0],
            {
                "impl_backend": "liger_tp",
                "chunk_size": 512,
                "tiles_per_reduce": 1,
            },
        )
    ]


def test_megatron_engine_rejects_non_bf16_liger_tp():
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(use_fused_kernels=True, use_remove_padding=True)
    engine.model_config = SimpleNamespace(
        fused_kernel_options={"impl_backend": "liger_tp"},
        mtp=SimpleNamespace(enable=False),
    )
    engine.is_value_model = False
    engine.param_dtype = torch.float16
    engine.module = [SimpleNamespace(post_process=False)]

    with pytest.raises(ValueError, match="requires Megatron model dtype bfloat16"):
        engine._maybe_enable_fused_kernels()
