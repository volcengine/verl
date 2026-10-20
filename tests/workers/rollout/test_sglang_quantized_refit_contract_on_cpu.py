# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
import torch

pytest.importorskip("compressed_tensors")
pytest.importorskip("sglang")

from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater

from verl.utils.qat.compressed_tensors import (
    CompressedTensorsWeightQuantizer,
    Fp8ConfigWeightQuantizer,
    build_compressed_tensors_config,
)
from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights):
        self.calls.append(list(weights))


def _exported_fp8_weights(native_fp8: bool):
    weight = torch.linspace(-1, 1, 256 * 256, dtype=torch.bfloat16).reshape(256, 256)
    if native_fp8:
        quantizer = Fp8ConfigWeightQuantizer(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        )
    else:
        config = build_compressed_tensors_config(
            linear="FP8_BLOCK",
            moe=None,
            ignore=[],
            config_file=None,
        )
        quantizer = CompressedTensorsWeightQuantizer(config)
    return list(quantizer([("model.layers.0.self_attn.q_proj.weight", weight)]))


@pytest.mark.parametrize(
    ("native_fp8", "scale_name"),
    [
        (False, "model.layers.0.self_attn.q_proj.weight_scale"),
        (True, "model.layers.0.self_attn.q_proj.weight_scale_inv"),
    ],
)
def test_sglang_tensor_update_preserves_quantized_export_names(monkeypatch, native_fp8, scale_name):
    exported = _exported_fp8_weights(native_fp8)
    model = _RecordingModel()
    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: None,
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.model_runner_components.weight_updater._unsupported_derived_weight_cache_error",
        lambda: None,
    )
    monkeypatch.setattr(WeightUpdater, "_assert_weight_cache_inactive", lambda self, op: None)

    ok, _ = updater.update_weights_from_tensor(exported)

    assert ok
    assert {name for name, _ in model.calls[0]} == {
        "model.layers.0.self_attn.q_proj.weight",
        scale_name,
    }


@pytest.mark.parametrize("native_fp8", [False, True])
def test_sglang_delta_loader_preserves_quantized_export_names(monkeypatch, native_fp8):
    exported = _exported_fp8_weights(native_fp8)
    model = _RecordingModel()
    monkeypatch.setattr(
        "verl.checkpoint_engine.delta_sync.encode.checksum",
        lambda positions, values: 0,
    )

    for name, tensor in exported:
        values = tensor.flatten()
        positions = torch.empty(0, dtype=torch.uint8)
        spec = {
            "encoding": "dense",
            "params": [
                {
                    "name": name,
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                    "pos_start": 0,
                    "pos_end": 0,
                    "pos_width": 4,
                    "val_start": 0,
                    "val_end": tensor.numel(),
                }
            ],
            "checksum": 0,
        }
        encoded = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
        apply_delta(
            model,
            [
                ("__delta_spec__", encoded),
                ("__positions__", positions),
                ("__values__", values),
            ],
        )

    assert [name for call in model.calls for name, _ in call] == [name for name, _ in exported]
