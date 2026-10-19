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

"""CPU tests for the DeepSeek-V4 branch of ``Fp8ConfigWeightQuantizer``.

AutoModel exports DeepSeek-V4 in BF16, so the trainer-side quantizer must
re-quantize it into the checkpoint's private FP8/FP4 layout, while the
Megatron-Bridge stream is already quantized and only needs a dtype reinterpret.
"""

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from verl.utils.qat.compressed_tensors import Fp8ConfigWeightQuantizer

_VLLM_UTILS_DIR = Path(__file__).resolve().parents[2] / "verl/utils/vllm"
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@contextmanager
def _verl_vllm_without_vllm():
    """Expose the ``verl/utils/vllm`` helpers without the package ``__init__``, which imports vLLM."""
    package = types.ModuleType("verl.utils.vllm")
    package.__path__ = [str(_VLLM_UTILS_DIR)]
    with patch.dict(sys.modules, {"verl.utils.vllm": package}):
        for name in [name for name in sys.modules if name.startswith("verl.utils.vllm.")]:
            del sys.modules[name]
        yield


def _dsv41_quantizer():
    return Fp8ConfigWeightQuantizer(
        {"quant_method": "fp8", "weight_block_size": [32, 32]}, model_types={"deepseek_v41", None}
    )


def _dequantize_mxfp4(packed, scale):
    codes = packed.view(torch.uint8)
    nibbles = torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten(-2)
    magnitude = _E2M1_GRID[(nibbles & 0x7).long()]
    values = torch.where((nibbles & 0x8) > 0, -magnitude, magnitude)
    return values * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)


def test_deepseek_v4_quantizer_requantizes_automodel_bf16_export():
    # Every 32-wide block spans the whole E2M1 grid, scaled by a per-row power of two.
    index = torch.arange(64 * 64).reshape(64, 64)
    sign = torch.where((index // 8) % 2 == 0, 1.0, -1.0)
    row_scale = torch.exp2((torch.arange(64) % 3 - 1).float())[:, None]
    expert = (_E2M1_GRID[index % 8] * sign * row_scale).to(torch.bfloat16)
    linear = (index % 31 - 15).to(torch.bfloat16)
    norm = torch.ones(64, dtype=torch.bfloat16)

    with _verl_vllm_without_vllm():
        result = dict(
            _dsv41_quantizer()(
                [
                    ("layers.0.ffn.experts.0.w1.weight", expert),
                    ("layers.0.attn.wq_a.weight", linear),
                    ("layers.0.attn_norm.weight", norm),
                ]
            )
        )

    assert set(result) == {
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.scale",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.attn_norm.weight",
    }
    packed, expert_scale = result["layers.0.ffn.experts.0.w1.weight"], result["layers.0.ffn.experts.0.w1.scale"]
    assert packed.dtype == torch.uint8 and packed.shape == (64, 32)
    assert expert_scale.dtype == torch.uint8 and expert_scale.shape == (64, 2)
    torch.testing.assert_close(_dequantize_mxfp4(packed, expert_scale), expert.float(), rtol=0, atol=0)

    fp8, linear_scale = result["layers.0.attn.wq_a.weight"], result["layers.0.attn.wq_a.scale"]
    assert fp8.dtype == torch.float8_e4m3fn
    # 32x32 blocks from ``weight_block_size``, not the 128x128 default.
    assert linear_scale.dtype == torch.float8_e8m0fnu and linear_scale.shape == (2, 2)
    block_scale = linear_scale.float().repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    torch.testing.assert_close(fp8.float() * block_scale, linear.float(), rtol=0, atol=0)

    assert result["layers.0.attn_norm.weight"] is norm


def test_deepseek_v4_quantizer_reinterprets_megatron_bridge_export():
    packed = torch.tensor([[-1, 2]], dtype=torch.int8)
    scale = torch.tensor([[127]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    fp8 = torch.zeros(32, 32, dtype=torch.float8_e4m3fn)

    with _verl_vllm_without_vllm():
        result = dict(
            _dsv41_quantizer()(
                [
                    ("layers.0.ffn.experts.0.w1.weight", packed),
                    ("layers.0.ffn.experts.0.w1.scale", scale),
                    ("layers.0.attn.wq_a.weight", fp8),
                ]
            )
        )

    for name, source in (("layers.0.ffn.experts.0.w1.weight", packed), ("layers.0.ffn.experts.0.w1.scale", scale)):
        assert result[name].dtype == torch.uint8
        assert result[name].data_ptr() == source.data_ptr()
    assert result["layers.0.attn.wq_a.weight"] is fp8


def test_deepseek_v4_quantizer_without_vllm_keeps_plain_passthrough():
    packed = torch.tensor([[-1, 2]], dtype=torch.int8)
    expert = torch.ones(64, 64, dtype=torch.bfloat16)

    # SGLang environments do not ship vLLM, so ``verl.utils.vllm`` cannot be imported.
    with patch.dict(sys.modules, {"verl.utils.vllm": None, "verl.utils.vllm.vllm_fp4_utils": None}):
        result = dict(
            _dsv41_quantizer()(
                [
                    ("layers.0.ffn.experts.0.w1.weight", packed),
                    ("layers.0.ffn.experts.1.w1.weight", expert),
                ]
            )
        )

    assert result["layers.0.ffn.experts.0.w1.weight"].dtype == torch.uint8
    assert result["layers.0.ffn.experts.1.w1.weight"] is expert


def test_deepseek_v4_quantizer_rejects_non_square_blocks():
    with pytest.raises(NotImplementedError, match="square FP8 weight blocks"):
        Fp8ConfigWeightQuantizer({"quant_method": "fp8", "weight_block_size": [32, 128]}, model_types={"deepseek_v4"})
