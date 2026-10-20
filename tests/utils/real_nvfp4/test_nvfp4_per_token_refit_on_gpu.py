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

"""NVFP4_PER_TOKEN rollout refit on a small Qwen3-MoE.

The preset engine refits from the trainer-side export through vLLM's layerwise reload.
Its greedy logprobs must equal vLLM's online nvfp4_per_token method loading the same BF16
weights, and with CUDA graphs a refit must restore them exactly, including after other
weights and after a level-2 sleep.
"""

import gc
import os

import pytest
import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
pytest.importorskip("compressed_tensors")
pytest.importorskip("vllm")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    NVFP4PerTokenWeightQuantizer,
    build_compressed_tensors_config,
    moe_layer_partition,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="NVFP4_PER_TOKEN rollout needs a Blackwell (SM100) GPU",
)

NUM_LAYERS = 3
PROMPTS = [[11, 523, 97, 1024, 7], [3, 3, 3, 900, 41, 2049], [1500, 8, 77]]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=4096,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=32,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        max_position_embeddings=1024,
        torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("qwen3_moe_small")
    model.save_pretrained(path)
    return str(path)


def _weights(model_dir, scale=1.0):
    from safetensors import safe_open

    out = []
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, name), "pt", device="cuda") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    out.append((key, tensor * scale if ".experts." in key else tensor))
    return out


def _engine(model_dir, eager, **kwargs):
    from vllm import LLM

    return LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.2,
        enforce_eager=eager,
        compilation_config=None if eager else {"cudagraph_mode": "FULL_DECODE_ONLY"},
        enable_flashinfer_autotune=False,
        seed=0,
        **kwargs,
    )


def _runner(llm):
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    return core.model_executor.driver_worker.worker.model_runner


def _refit(llm, weights):
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

    runner = _runner(llm)
    model = runner.get_model()
    initialize_layerwise_reload(model)
    for start in range(0, len(weights), 32):
        model.load_weights([(name, tensor.clone()) for name, tensor in weights[start : start + 32]])
    finalize_layerwise_reload(model, runner.vllm_config.model_config)


def _logprobs(llm):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=0)
    outputs = llm.generate([{"prompt_token_ids": p} for p in PROMPTS], params, use_tqdm=False)
    return [[next(iter(lp.values())).logprob for lp in o.outputs[0].logprobs] for o in outputs]


def _kv_caches(llm):
    from vllm.model_executor.layers.attention import Attention

    caches = []
    for module in _runner(llm).get_model().modules():
        if isinstance(module, Attention):
            kv = module.kv_cache
            caches.extend(t for t in (kv if isinstance(kv, list | tuple) else [kv]) if isinstance(t, torch.Tensor))
    return caches


def _close(llm):
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def test_preset_refit_matches_online_quantization(model_dir):
    from verl.utils.vllm.vllm_nvfp4_per_token_utils import attest_nvfp4_per_token_runtime, clear_kv_cache
    from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches

    carved = [0, NUM_LAYERS - 1]
    # vLLM's online method quantizes the BF16 checkpoint at load. Eager mode keeps the
    # comparison independent of each engine's KV-cache size, which steers graph kernels.
    online = _engine(
        model_dir,
        eager=True,
        quantization="nvfp4_per_token",
        quantization_config={"ignore": [f"model.layers.{i}.mlp.experts" for i in carved]},
        moe_backend="flashinfer_trtllm",
    )
    expected = _logprobs(online)
    _close(online)

    apply_vllm_quant_patches()
    ignore = [r"re:.*layers\.(" + "|".join(map(str, carved)) + r")\.mlp\.experts\..*"]
    config = build_compressed_tensors_config(linear=None, moe="NVFP4_PER_TOKEN", ignore=ignore, config_file=None)

    def refit(llm, scale):
        runner = _runner(llm)
        quantized, bf16 = moe_layer_partition(runner.vllm_config.model_config.hf_config, config)
        assert (quantized, bf16) == ([1], carved)
        _refit(llm, list(NVFP4PerTokenWeightQuantizer(config)(_weights(model_dir, scale))))
        attest_nvfp4_per_token_runtime(runner.get_model(), quantized_layers=quantized, bf16_layers=bf16)
        return _logprobs(llm)

    preset = _engine(model_dir, eager=True, load_format="dummy", hf_overrides={"quantization_config": config})
    # Start-up forwards with the dummy weights can leave NaN in the KV cache; the rollout
    # clears it once the engine is up (monkey_patch_model).
    clear_kv_cache(_runner(preset).get_model())
    assert _kv_caches(preset) and all(torch.isfinite(t).all() for t in _kv_caches(preset))
    assert refit(preset, 1.0) == expected
    _close(preset)

    preset = _engine(
        model_dir,
        eager=False,
        load_format="dummy",
        enable_sleep_mode=True,
        hf_overrides={"quantization_config": config},
    )
    first = refit(preset, 1.0)
    assert refit(preset, 1.5) != first
    assert refit(preset, 1.0) == first
    # A level-2 sleep discards the weights; the next refit must restore every captured tensor.
    preset.sleep(level=2)
    preset.wake_up()
    assert refit(preset, 1.0) == first
    _close(preset)
