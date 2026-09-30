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

from verl.models.mcore import model_forward_fused as mff
from verl.utils.kernel import linear_cross_entropy
from verl.workers.engine.megatron import transformer_impl


@pytest.fixture
def engine_config():
    return SimpleNamespace(
        use_fused_kernels=True,
        use_remove_padding=True,
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=2048,
        context_parallel_size=2,
    )


def _model(post_process, tied=False, dtype=torch.bfloat16):
    # All PP/VPP chunks retain GPTModel's global dimensions. Do not infer
    # configuration from an output weight that only the final stage owns.
    return SimpleNamespace(
        post_process=post_process,
        share_embeddings_and_output_weights=tied,
        vocab_size=124160,
        config=SimpleNamespace(hidden_size=5120, params_dtype=dtype),
        parameters=lambda: iter([torch.empty(1)]),
    )


@pytest.mark.parametrize("post_process", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_configure_every_pipeline_stage_from_model_dimensions(monkeypatch, engine_config, post_process, tied):
    model = _model(post_process, tied)
    process_group = object()
    calls = []
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: process_group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(linear_cross_entropy, "configure_liger_tp_flsce", lambda **kwargs: calls.append(kwargs) or True)

    assert mff._configure_liger_tp_runtime(model, engine_config) is True
    assert calls == [
        {
            "max_tokens": 8192,
            "hidden_size": 5120,
            "local_vocab_size": 62080,
            "process_group": process_group,
            "device": torch.device("cpu"),
        }
    ]


def test_configuration_requires_token_capacity(engine_config):
    engine_config.max_token_len_per_gpu = None
    engine_config.infer_max_token_len_per_gpu = None
    with pytest.raises(RuntimeError, match="max-token limit"):
        mff._configure_liger_tp_runtime(_model(False), engine_config)


@pytest.mark.parametrize("use_liger", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("post_process", [False, True])
def test_patch_owns_backend_selection_and_runtime_setup(monkeypatch, engine_config, use_liger, dtype, post_process):
    model = _model(post_process, dtype=dtype)
    calls = []
    monkeypatch.setattr(mff, "_get_patching_model", lambda value: value)
    monkeypatch.setattr(mff, "_resolve_fused_forward_mode", lambda value: mff._HOOK_MODE)
    monkeypatch.setattr(mff, "_configure_liger_tp_runtime", lambda model, config: calls.append((model, config)) or True)
    mff.patch_fused_forward(model, SimpleNamespace(use_liger=use_liger), engine_config=engine_config)
    assert getattr(model, mff._FUSED_IMPL_BACKEND_ATTR) == ("liger" if use_liger else "triton")
    assert calls == ([(model, engine_config)] if use_liger and dtype == torch.bfloat16 else [])


def test_engine_delegates_all_virtual_chunks(monkeypatch, engine_config):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = engine_config
    engine.model_config = SimpleNamespace(use_liger=True, mtp=SimpleNamespace(enable=False))
    engine.is_value_model = False
    engine.module = [_model(False), _model(True)]
    calls = []
    monkeypatch.setattr(
        mff, "patch_fused_forward", lambda model, config, **kwargs: calls.append((model, config, kwargs))
    )
    engine._maybe_enable_fused_kernels()
    assert calls == [(model, engine.model_config, {"engine_config": engine_config}) for model in engine.module]
