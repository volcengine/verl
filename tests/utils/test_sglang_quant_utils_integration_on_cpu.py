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

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("compressed_tensors")
pytest.importorskip("sglang")

from verl.utils.qat.compressed_tensors import (
    CompressedTensorsWeightQuantizer,
    build_compressed_tensors_config,
    dispatch_rollout_weight_quantizer,
)
from verl.utils.sglang.sglang_quant_utils import (
    LOADER_FQN,
    SGLangMxfp8WeightQuantizer,
    enforce_sglang_deep_gemm,
    prepare_sglang_quantization,
    resolve_sglang_quantization_config,
    translate_mxfp8_to_sglang_fp8_config,
    uses_sglang_refit_loader,
)


def _config(linear="MXFP8", moe=None):
    return build_compressed_tensors_config(
        linear=linear,
        moe=moe,
        ignore=[r"re:.*linear_attn\.in_proj_[ab]$"],
        config_file=None,
    )


@pytest.mark.parametrize("moe", [None, "MXFP8"])
def test_all_mxfp8_translation_uses_native_sglang_schema(moe):
    ct_config = _config(moe=moe)

    native = translate_mxfp8_to_sglang_fp8_config(ct_config)

    assert native == {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "ignored_layers": [
            "lm_head",
            "embed_tokens",
            "visual",
            "mtp",
            "mlp.gate",
            "shared_expert_gate",
            "in_proj_ba",
            "in_proj_b",
            "in_proj_a",
            *(["mlp.experts"] if moe is None else []),
        ],
    }


def test_mixed_scheme_translation_is_rejected():
    with pytest.raises(ValueError, match="every config group"):
        translate_mxfp8_to_sglang_fp8_config(_config(moe="FP8_BLOCK"))


def test_native_sglang_config_honors_ct_regex_ignores():
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.utils import is_layer_skipped

    native = translate_mxfp8_to_sglang_fp8_config(_config(moe=None))
    config = Fp8Config.from_config(native)

    assert is_layer_skipped("model.layers.0.linear_attn.in_proj_ba", config.ignored_layers)
    assert is_layer_skipped("model.layers.0.linear_attn.in_proj_b", config.ignored_layers)
    assert is_layer_skipped("model.layers.0.linear_attn.in_proj_a", config.ignored_layers)
    assert is_layer_skipped("model.layers.0.mlp.experts", config.ignored_layers)
    assert not is_layer_skipped("model.layers.0.self_attn.q_proj", config.ignored_layers)


def test_unknown_regex_ignore_is_rejected_before_sglang_launch():
    config = _config()
    config["ignore"].append(r"re:.*custom_proj$")
    with pytest.raises(ValueError, match="does not support regex"):
        translate_mxfp8_to_sglang_fp8_config(config)


def test_sglang_export_renames_only_the_ct_mxfp8_scale():
    ct_config = _config(moe="MXFP8")
    weight = torch.linspace(-1, 1, 64 * 64, dtype=torch.bfloat16).reshape(64, 64)
    name = "model.layers.0.self_attn.q_proj.weight"
    ct = dict(CompressedTensorsWeightQuantizer(ct_config)([(name, weight)]))
    native = dict(SGLangMxfp8WeightQuantizer(ct_config)([(name, weight)]))

    assert native.keys() == {name, name + "_scale_inv"}
    assert torch.equal(native[name], ct[name])
    assert torch.equal(native[name + "_scale_inv"], ct[name + "_scale"])


def test_dispatch_selects_native_names_for_sglang_override_and_model_config():
    model = SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(model_type="qwen", text_config=None, quantization_config=None),
    )

    def rollout(name):
        return SimpleNamespace(
            name=name,
            quantization=SimpleNamespace(linear="MXFP8", moe=None, ignore=[], config_file=None),
        )

    assert isinstance(
        dispatch_rollout_weight_quantizer(model_config=model, rollout_config=rollout("sglang")),
        SGLangMxfp8WeightQuantizer,
    )
    assert isinstance(
        dispatch_rollout_weight_quantizer(model_config=model, rollout_config=rollout("vllm")),
        CompressedTensorsWeightQuantizer,
    )

    model.hf_config.quantization_config = _config()
    native_rollout = SimpleNamespace(
        name="sglang",
        quantization=SimpleNamespace(linear=None, moe=None, ignore=[], config_file=None),
    )
    assert isinstance(
        dispatch_rollout_weight_quantizer(model_config=model, rollout_config=native_rollout),
        SGLangMxfp8WeightQuantizer,
    )


def test_resolve_sglang_quantization_config_prefers_override_and_falls_back_to_native():
    native = _config(moe="MXFP4")
    model = SimpleNamespace(
        local_path="/model",
        hf_config=SimpleNamespace(
            quantization_config=None,
            text_config=SimpleNamespace(quantization_config=SimpleNamespace(to_dict=lambda: native)),
        ),
    )
    disabled = SimpleNamespace(quantization=SimpleNamespace(enabled=False))

    resolved = resolve_sglang_quantization_config(disabled, model)
    assert resolved == native
    assert resolved is not None
    assert uses_sglang_refit_loader(resolved)

    override = _config(moe="MXFP8")
    enabled = SimpleNamespace(
        quantization=SimpleNamespace(
            enabled=True,
            build_quantization_config=lambda model_path: override,
        )
    )
    assert resolve_sglang_quantization_config(enabled, model) is override


def test_sglang_launch_translation_registers_loader_and_rejects_unsafe_backend(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    loaders = []
    engine_kwargs = {}
    native = prepare_sglang_quantization(_config(moe="MXFP8"), loaders, engine_kwargs)
    assert native["quant_method"] == "mxfp8"
    assert loaders == [LOADER_FQN]
    assert engine_kwargs == {
        "fp8_gemm_runner_backend": "deep_gemm",
        "moe_runner_backend": "deep_gemm",
    }
    assert "in_proj_ba" in os.environ["SGLANG_FP8_IGNORED_LAYERS"].split(",")

    with pytest.raises(ValueError, match="requires fp8_gemm_runner_backend=deep_gemm"):
        prepare_sglang_quantization(
            _config(),
            [],
            {"fp8_gemm_runner_backend": "flashinfer_trtllm"},
        )


def test_sglang_mxfp4_moe_registers_refit_loader_and_pins_deep_gemm_backend():
    config = build_compressed_tensors_config(
        linear=None,
        moe="MXFP4",
        ignore=[],
        config_file=None,
    )
    loaders, engine_kwargs = [], {}

    native = prepare_sglang_quantization(config, loaders, engine_kwargs)
    assert native is not config
    assert r"re:.*" in native["ignore"] and r"re:.*" not in config["ignore"]
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

    parsed = CompressedTensorsConfig.from_config(native)
    assert parsed.get_linear_scheme(torch.nn.Linear(8, 8), "model.layers.0.self_attn.q_proj") is None
    assert parsed._is_mxfp4_moe("model.layers.0.mlp.experts")
    assert loaders == [LOADER_FQN]
    assert engine_kwargs == {
        "fp8_gemm_runner_backend": "deep_gemm",
        "moe_runner_backend": "deep_gemm",
    }

    explicit = {"moe_runner_backend": "deep_gemm"}
    prepare_sglang_quantization(config, [], explicit)
    assert explicit["moe_runner_backend"] == "deep_gemm"

    with pytest.raises(ValueError, match="requires moe_runner_backend=deep_gemm"):
        prepare_sglang_quantization(config, [], {"moe_runner_backend": "flashinfer_mxfp4"})


def test_sglang_nvfp4_moe_registers_refit_loader_and_leaves_backend_validation_to_sglang():
    config = build_compressed_tensors_config(
        linear=None,
        moe="NVFP4",
        ignore=[],
        config_file=None,
    )
    loaders, engine_kwargs = [], {}

    native = prepare_sglang_quantization(config, loaders, engine_kwargs)
    assert native is not config
    assert r"re:^(?!.*mlp\.experts(?:\.|$)).*$" in native["ignore"]
    assert loaders == [LOADER_FQN]
    assert engine_kwargs == {}

    explicit = {"moe_runner_backend": "flashinfer_trtllm"}
    prepare_sglang_quantization(config, [], explicit)
    assert explicit == {"moe_runner_backend": "flashinfer_trtllm"}


def test_deep_gemm_is_forced_for_fp8_override_and_native_hf_configs_but_not_nvfp4():
    fp8_config = build_compressed_tensors_config(
        linear="FP8_BLOCK",
        moe="FP8_BLOCK",
        ignore=[],
        config_file=None,
    )
    kwargs = {"fp8_gemm_runner_backend": "auto", "moe_runner_backend": "auto"}
    assert prepare_sglang_quantization(fp8_config, [], kwargs) is fp8_config
    assert kwargs == {
        "fp8_gemm_runner_backend": "deep_gemm",
        "moe_runner_backend": "deep_gemm",
    }

    for hf_config in (
        {"quant_method": "fp8"},
        {"quant_method": "mxfp8"},
        {"quant_method": "modelopt", "quantization": {"quant_algo": "FP8"}},
    ):
        kwargs = {}
        enforce_sglang_deep_gemm(hf_config, kwargs)
        assert kwargs == {
            "fp8_gemm_runner_backend": "deep_gemm",
            "moe_runner_backend": "deep_gemm",
        }

    nvfp4_config = build_compressed_tensors_config(
        linear=None,
        moe="NVFP4",
        ignore=[],
        config_file=None,
    )
    kwargs = {"fp8_gemm_runner_backend": "flashinfer_cutlass", "moe_runner_backend": "flashinfer_trtllm"}
    enforce_sglang_deep_gemm(nvfp4_config, kwargs)
    assert kwargs == {
        "fp8_gemm_runner_backend": "flashinfer_cutlass",
        "moe_runner_backend": "flashinfer_trtllm",
    }
