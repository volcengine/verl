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
"""CPU tests for the SGLang quantized refit loader on MXFP8 layers (sglang stubbed)."""

import sys
import types
from typing import NamedTuple

import pytest
import torch

from verl.utils.sglang import sglang_fp8_utils as fp8_utils
from verl.utils.sglang import sglang_quant_utils as quant_utils


class _Backend:
    def __init__(self, name):
        self.name = name

    def is_flashinfer_trtllm(self):
        return self.name == "flashinfer_trtllm"

    def is_flashinfer_cutlass(self):
        return self.name == "flashinfer_cutlass"

    def is_flashinfer_cutedsl(self):
        return self.name == "flashinfer_cutedsl"

    def is_deep_gemm(self):
        return self.name == "deep_gemm"


class _StandardTopKOutput(NamedTuple):  # sglang.srt.layers.moe.topk.StandardTopKOutput
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


# The sglang stubs are installed through the running test's monkeypatch, so they are removed again when the
# test ends instead of leaking into later tests in the same process.
_STUB_PATCHES: list = []


@pytest.fixture(autouse=True)
def _undo_sglang_stubs(monkeypatch):
    _STUB_PATCHES.append(monkeypatch)
    yield
    _STUB_PATCHES.pop()


def _set_module(name, module):
    _STUB_PATCHES[-1].setitem(sys.modules, name, module)


def _install_stub(backend_name):
    mod = types.ModuleType("sglang.srt.layers.quantization.fp8_utils")
    mod.get_fp8_gemm_runner_backend = lambda: _Backend(backend_name)
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.moe",
    ):
        if name not in sys.modules:
            _set_module(name, types.ModuleType(name))
    _set_module("sglang.srt.layers.quantization.fp8_utils", mod)
    topk = types.ModuleType("sglang.srt.layers.moe.topk")
    topk.StandardTopKOutput = _StandardTopKOutput
    _set_module("sglang.srt.layers.moe.topk", topk)


class _QuantMethod:
    use_mxfp8 = True
    is_checkpoint_fp8_serialized = True

    def __init__(self, resolved_backend=None):
        self.calls = 0
        # sglang >= 0.5.18 stores the resolved MXFP8 dense backend on the quant method
        if resolved_backend is not None:
            self.mxfp8_dense_backend = _Backend(resolved_backend)

    def process_weights_after_loading(self, layer):
        self.calls += 1
        layer.weight_scale_inv_swizzled = layer.weight_scale_inv.clone() + 1


class _Linear(torch.nn.Module):
    def __init__(self, resolved_backend=None):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randint(-6, 7, (4, 64), generator=g).to(torch.float8_e4m3fn), requires_grad=False
        )
        self.weight_scale_inv = torch.nn.Parameter(torch.full((4, 2), 127, dtype=torch.uint8), requires_grad=False)
        self.quant_method = _QuantMethod(resolved_backend)
        self.weight_scale_inv_swizzled = self.weight_scale_inv.clone() + 1  # as after the initial load


class _MoEQuantMethod:
    """sglang ``Fp8MoEMethod`` with an fp8-serialized MXFP8 checkpoint on the CUTLASS MoE runner of sglang >= 0.5.16:
    post-load processing swizzles the expert scales *in place* into a different shape."""

    use_mxfp8 = True
    is_checkpoint_fp8_serialized = True

    def __init__(self):
        self.calls = 0

    def process_weights_after_loading(self, layer):
        self.calls += 1
        # the copy the kernel reads is derived from the canonical scales present at this moment
        layer._kernel_scales = (layer.w13_weight_scale_inv.data.clone(), layer.w2_weight_scale_inv.data.clone())
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            p = getattr(layer, name)
            swizzled = p.data.reshape(p.data.shape[0], -1).clone() + 1
            if p.data.shape == swizzled.shape:
                p.data.copy_(swizzled)
            else:
                p.data = swizzled  # sglang's _copy_or_rebind on shape change


class _MoE(torch.nn.Module):
    def __init__(self, experts=2, inter=64, hidden=64, random=False):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        w13 = (
            torch.randint(-6, 7, (experts, 2 * inter, hidden), generator=g)
            if random
            else torch.zeros(experts, 2 * inter, hidden)
        )
        w2 = (
            torch.randint(-6, 7, (experts, hidden, inter), generator=g)
            if random
            else torch.zeros(experts, hidden, inter)
        )
        self.w13_weight = torch.nn.Parameter(w13.to(torch.float8_e4m3fn), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2.to(torch.float8_e4m3fn), requires_grad=False)
        self.moe_ep_size = 1
        self.moe_tp_size = 1
        self.reduce_results = False
        self.num_experts = experts
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.full((experts, 2 * inter, hidden // 32), 127, dtype=torch.uint8), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.full((experts, hidden, inter // 32), 127, dtype=torch.uint8), requires_grad=False
        )
        self.quant_method = _MoEQuantMethod()


class _Model(torch.nn.Module):
    def __init__(self, resolved_backend=None):
        super().__init__()
        self.a = _Linear(resolved_backend)
        self.b = _Linear(resolved_backend)
        self.plain = torch.nn.Linear(2, 2)  # no quant_method → ignored
        self.loaded = []

    def load_weights(self, named_tensors):
        for name, t in named_tensors:
            self.loaded.append(name)
            module_name, param = name.rsplit(".", 1)
            getattr(getattr(self, module_name), param).data.copy_(t)


def test_cutlass_backend_reprocesses_every_mxfp8_layer_after_load():
    _install_stub("flashinfer_cutlass")
    m = _Model()
    new_scale = torch.full((4, 2), 7, dtype=torch.uint8)
    quant_utils.load_and_reprocess(
        m, [("a.weight_scale_inv", new_scale), ("b.weight_scale_inv", new_scale)] + quant_utils.finalize_tensors()
    )
    assert m.loaded == ["a.weight_scale_inv", "b.weight_scale_inv"]
    assert m.a.quant_method.calls == 1 and m.b.quant_method.calls == 1
    # swizzled copy derived from the *new* canonical scales
    assert torch.equal(m.a.weight_scale_inv_swizzled, new_scale + 1)


def test_triton_backend_is_a_noop_after_load():
    _install_stub("triton")
    m = _Model()
    n = fp8_utils.reprocess_mxfp8_layers(m)
    assert n == 0 and m.a.quant_method.calls == 0


def test_trtllm_backend_is_rejected():
    _install_stub("flashinfer_trtllm")
    m = _Model()
    try:
        fp8_utils.reprocess_mxfp8_layers(m)
    except NotImplementedError as e:
        assert "flashinfer_trtllm" in str(e)
    else:
        raise AssertionError("expected NotImplementedError")


def test_resolved_backend_on_layer_wins_over_launch_flag():
    # sglang >= 0.5.18: --fp8-gemm-backend left at auto, but the quant method resolved the
    # MXFP8 dense backend to FlashInfer CuTe-DSL (the Blackwell default). The derived copy
    # must be rebuilt even though the requested backend is not "flashinfer_cutlass".
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutedsl")
    new_scale = torch.full((4, 2), 3, dtype=torch.uint8)
    quant_utils.load_and_reprocess(m, [("a.weight_scale_inv", new_scale)] + quant_utils.finalize_tensors())
    assert m.a.quant_method.calls == 1 and m.b.quant_method.calls == 1
    assert torch.equal(m.a.weight_scale_inv_swizzled, new_scale + 1)


def test_resolved_deep_gemm_backend_is_reprocessed():
    _install_stub("auto")
    m = _Model(resolved_backend="deep_gemm")
    assert fp8_utils.reprocess_mxfp8_layers(m) == 2


def test_resolved_trtllm_backend_is_rejected():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_trtllm")
    try:
        fp8_utils.reprocess_mxfp8_layers(m)
    except NotImplementedError as e:
        assert "flashinfer_trtllm" in str(e)
    else:
        raise AssertionError("expected NotImplementedError")


def test_moe_expert_scales_are_staged_reprocessed_and_folded_into_live_storage():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load: scales now swizzled [E, 256]
    assert tuple(m.moe.w13_weight_scale_inv.shape) == (2, 256)
    live_ptr = m.moe.w13_weight_scale_inv.data_ptr()

    new_w13 = torch.full((2, 128, 2), 7, dtype=torch.uint8)  # canonical [E, N, K/32] as the sync sends it
    new_w2 = torch.full((2, 64, 2), 5, dtype=torch.uint8)
    quant_utils.load_and_reprocess(
        m, [("moe.w13_weight_scale_inv", new_w13), ("moe.w2_weight_scale_inv", new_w2)] + quant_utils.finalize_tensors()
    )

    # load_weights could only write canonical tensors because the scales were staged back to that layout
    assert m.moe.quant_method.calls == 2
    p = m.moe.w13_weight_scale_inv
    assert tuple(p.shape) == (2, 256) and p.data_ptr() == live_ptr  # CUDA-graph storage kept
    assert torch.equal(p.data, torch.full((2, 256), 8, dtype=torch.uint8))  # re-swizzled from the NEW scales
    assert torch.equal(m.moe.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))


def test_moe_scales_whose_shape_the_runner_keeps_are_not_staged():
    class _CanonicalMoEQuantMethod(_MoEQuantMethod):
        # keeps the scales' shape and dtype, as the in-place runners do (FlashInfer TRT-LLM; CUTLASS on
        # sglang <= 0.5.15); staging is decided by shape and dtype, so the bytes left here do not matter
        def process_weights_after_loading(self, layer):
            self.calls += 1

    moe = _MoE()
    moe.quant_method = _CanonicalMoEQuantMethod()
    m = torch.nn.Module()
    m.moe = moe
    assert fp8_utils.stage_mxfp8_moe_scales(m) == []
    assert fp8_utils.reprocess_mxfp8_moe_layers(m, []) == 1


def _canonical_moe_model():
    """A model whose MoE runner keeps scales canonical (no staging) but derives a kernel copy at load."""

    class _CanonicalRunner(_MoEQuantMethod):
        def process_weights_after_loading(self, layer):
            self.calls += 1
            layer._kernel_scales = (layer.w13_weight_scale_inv.data.clone(), layer.w2_weight_scale_inv.data.clone())

    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE(random=True)
    m.moe.quant_method = _CanonicalRunner()
    m.moe.quant_method.process_weights_after_loading(m.moe)  # initial load
    return m


def _two_moe_model():
    """Two MoE layers on the swizzling runner after their initial load, next to the two dense layers."""
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe1, m.moe2 = _MoE(), _MoE()
    for moe in (m.moe1, m.moe2):
        moe.quant_method.process_weights_after_loading(moe)  # initial load: scales swizzled in place
    return m


def _moe_scales(name, w13=7, w2=5):
    return [
        (f"{name}.w13_weight_scale_inv", torch.full((2, 128, 2), w13, dtype=torch.uint8)),
        (f"{name}.w2_weight_scale_inv", torch.full((2, 64, 2), w2, dtype=torch.uint8)),
    ]


def test_a_sync_split_across_requests_re_derives_every_layer_once_after_the_last_bucket():
    # One sync arrives as one request per bucket. Re-deriving per request rebuilt moe2 from its still-0xFF
    # staging buffer during request 1 and folded that into its live buffer.
    m = _two_moe_model()
    live = {name: getattr(m, name).w13_weight_scale_inv.data for name in ("moe1", "moe2")}
    dense = [("a.weight_scale_inv", torch.full((4, 2), 7, dtype=torch.uint8))]
    quant_utils.load_and_reprocess(m, _moe_scales("moe1") + dense)
    quant_utils.load_and_reprocess(m, _moe_scales("moe2"))
    # nothing is re-derived before the closing request, dense layers included
    assert m.moe1.quant_method.calls == m.moe2.quant_method.calls == 1 and m.a.quant_method.calls == 0

    quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    for name in ("moe1", "moe2"):
        moe = getattr(m, name)
        assert moe.quant_method.calls == 2
        assert moe.w13_weight_scale_inv.data_ptr() == live[name].data_ptr()  # CUDA-graph storage kept
        assert torch.equal(moe.w13_weight_scale_inv.data, torch.full((2, 256), 8, dtype=torch.uint8))
        assert torch.equal(moe.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))
    assert m.a.quant_method.calls == 1
    assert torch.equal(m.a.weight_scale_inv_swizzled, torch.full((4, 2), 8, dtype=torch.uint8))


def test_a_moe_layer_whose_scales_span_two_requests_is_re_derived_from_both():
    m = _two_moe_model()
    w13, w2 = _moe_scales("moe1")
    quant_utils.load_and_reprocess(m, [w13] + _moe_scales("moe2"))
    quant_utils.load_and_reprocess(m, [w2] + quant_utils.finalize_tensors())
    assert torch.equal(m.moe1.w13_weight_scale_inv.data, torch.full((2, 256), 8, dtype=torch.uint8))
    assert torch.equal(m.moe1.w2_weight_scale_inv.data, torch.full((2, 128), 6, dtype=torch.uint8))


def test_closing_request_without_a_loaded_bucket_re_derives_nothing():
    m = _two_moe_model()
    quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    # re-running the processing on already-derived layouts would apply it twice
    assert m.moe1.quant_method.calls == 1 and m.a.quant_method.calls == 0


def test_each_sync_stages_once_and_re_derives_once():
    m = _two_moe_model()
    for sync, value in enumerate((7, 9), start=1):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1", w13=value) + _moe_scales("moe2", w13=value))
        quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
        assert m.moe1.quant_method.calls == 1 + sync
        assert torch.equal(m.moe1.w13_weight_scale_inv.data, torch.full((2, 256), value + 1, dtype=torch.uint8))


def _live_scales(m):
    """The storage each expert scale had after the initial load: what the CUDA graph captured."""
    return {
        (name, sname): getattr(getattr(m, name), sname).data
        for name in ("moe1", "moe2")
        for sname in ("w13_weight_scale_inv", "w2_weight_scale_inv")
    }


def _assert_scales_on(m, live):
    for (name, sname), storage in live.items():
        assert getattr(getattr(m, name), sname).data_ptr() == storage.data_ptr(), f"{name}.{sname}"


def _assert_next_sync_folds_into(m, live, value=9):
    quant_utils.load_and_reprocess(m, _moe_scales("moe1", w13=value) + _moe_scales("moe2", w13=value))
    quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    _assert_scales_on(m, live)
    for name in ("moe1", "moe2"):
        expected = torch.full((2, 256), value + 1, dtype=torch.uint8)
        assert torch.equal(getattr(m, name).w13_weight_scale_inv.data, expected)


def test_a_bucket_that_fails_to_load_ends_the_sync_with_every_scale_on_its_live_storage():
    m = _two_moe_model()
    live = _live_scales(m)
    quant_utils.load_and_reprocess(m, _moe_scales("moe1"))  # the first request stages every expert scale
    with pytest.raises(RuntimeError):  # load_weights cannot copy a [3, 3] tensor into the scale
        quant_utils.load_and_reprocess(m, [("moe2.w13_weight_scale_inv", torch.zeros(3, 3, dtype=torch.uint8))])
    _assert_scales_on(m, live)
    assert quant_utils._SYNC_FP8_ATTR not in m.__dict__
    _assert_next_sync_folds_into(m, live)


def test_a_closing_request_that_fails_leaves_no_scale_on_its_staging_buffer():
    # The closing request used to drop the sync marker first. When the re-derivation then raised, moe2 stayed
    # on its staging buffer, whose canonical shape made the next sync skip staging it and re-derive it into a
    # new tensor, while the CUDA graph kept reading the live one.
    m = _two_moe_model()
    live = _live_scales(m)
    quant_utils.load_and_reprocess(m, _moe_scales("moe1") + _moe_scales("moe2"))

    def fail(layer):
        raise RuntimeError("post-load processing failed")

    m.moe2.quant_method.process_weights_after_loading = fail
    with pytest.raises(RuntimeError, match="post-load processing failed"):
        quant_utils.load_and_reprocess(m, quant_utils.finalize_tensors())
    del m.moe2.quant_method.process_weights_after_loading
    _assert_scales_on(m, live)
    assert quant_utils._SYNC_FP8_ATTR not in m.__dict__
    _assert_next_sync_folds_into(m, live)


def test_staging_rebinds_nothing_when_a_staging_buffer_cannot_be_allocated(monkeypatch):
    m = _two_moe_model()
    live = _live_scales(m)
    empty, allocated = torch.empty, []

    def empty_or_oom(*args, **kwargs):
        if len(allocated) == 2:  # the third staging buffer, moe2's w13 scale
            raise RuntimeError("CUDA out of memory")
        allocated.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_or_oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1"))
    monkeypatch.setattr(torch, "empty", empty)
    _assert_scales_on(m, live)  # moe1's two scales were not left on buffers no list records
    assert quant_utils._SYNC_FP8_ATTR not in m.__dict__
    _assert_next_sync_folds_into(m, live)


def test_reprocess_reports_a_staged_scale_that_was_not_folded_back():
    _install_stub("auto")
    m = _Model(resolved_backend="flashinfer_cutlass")
    m.moe = _MoE()
    m.moe.quant_method.process_weights_after_loading(m.moe)
    staged = fp8_utils.stage_mxfp8_moe_scales(m)
    assert len(staged) == 2  # two scale params of one module: the count is per parameter, not per module
    orphan = _MoE()  # staged entry whose module is not in the model -> never visited by reprocess
    orphan.quant_method.process_weights_after_loading(orphan)
    staged += fp8_utils.stage_mxfp8_moe_scales(torch.nn.ModuleDict({"x": orphan}))
    for module, sname, _ in staged:
        getattr(module, sname).data.fill_(7)  # "the sync wrote every scale", so only the fold-back check can fire
    try:
        fp8_utils.reprocess_mxfp8_moe_layers(m, staged)
    except RuntimeError as e:
        assert "not folded back into its live storage" in str(e)
    else:
        raise AssertionError("expected the post-condition to fire for the orphaned staged scale")


def test_loader_fqn_matches_function():
    mod, fn = quant_utils.REFIT_LOADER_FQN.rsplit(".", 1)
    assert mod == quant_utils.__name__ and getattr(quant_utils, fn) is quant_utils.load_and_reprocess


class UnquantizedFusedMoEMethod:  # matched by class name, as sglang's
    def __init__(self, trtllm=False):
        self.use_flashinfer_trtllm_moe = trtllm


class _BF16MoE(torch.nn.Module):
    """sglang ``FusedMoE`` under ``UnquantizedFusedMoEMethod``: a MoE layer ``ignored_layers`` keeps in BF16."""

    def __init__(self, experts=2, inter=64, hidden=64, trtllm=False, w13_shape=None):
        super().__init__()
        self.num_local_experts, self.intermediate_size_per_partition, self.hidden_size = experts, inter, hidden
        self.moe_runner_config = types.SimpleNamespace(is_gated=True, routed_scaling_factor=None)
        self.quant_method = UnquantizedFusedMoEMethod(trtllm)
        shape = w13_shape or (experts, 2 * inter, hidden)
        self.w13_weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
        w2 = torch.zeros(experts, hidden, inter, dtype=torch.bfloat16)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)


def test_a_bf16_moe_layer_on_the_trtllm_runner_is_refused_before_anything_is_written():
    # B200, sglang 0.5.20, MoE runner left to sglang: ignore=["model.layers.0"] on a Qwen3-MoE put layer 0's BF16
    # experts on the TRT-LLM runner, which sglang picks for every MoE layer of an MXFP8 model; the first sync then
    # failed inside FusedMoE._load_w13 on the block layout ("size of tensor a (64) must match ... (512)").
    m = _two_moe_model()
    m.moe0 = _BF16MoE(trtllm=True)
    with pytest.raises(NotImplementedError, match=r"moe0 \(w13 \(2, 128, 64\) bfloat16, FlashInfer TRT-LLM runner\)"):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1"))
    assert m.loaded == [] and quant_utils._SYNC_FP8_ATTR not in m.__dict__  # nothing written, the sync abandoned


def test_a_bf16_moe_layer_the_runner_re_laid_out_is_refused_on_any_runner():
    m = _two_moe_model()
    m.moe0 = _BF16MoE(w13_shape=(2, 128, 8, 8))
    with pytest.raises(NotImplementedError, match=r"checkpoint layout \(2, 128, 64\)"):
        quant_utils.load_and_reprocess(m, _moe_scales("moe1"))


def test_a_bf16_moe_layer_in_checkpoint_layout_is_synced():
    m = _two_moe_model()
    m.moe0 = _BF16MoE()
    assert fp8_utils.relaid_bf16_moe_layers(m) == []
    quant_utils.load_and_reprocess(m, _moe_scales("moe1"))
    assert m.loaded == ["moe1.w13_weight_scale_inv", "moe1.w2_weight_scale_inv"]


def test_quantized_moe_layers_are_not_taken_for_bf16_ones():
    class Mxfp4MoEMethod:  # packed FP4 experts do not have the BF16 checkpoint shape
        pass

    m = _two_moe_model()
    m.fp4 = _BF16MoE(w13_shape=(2, 128, 32))
    m.fp4.quant_method = Mxfp4MoEMethod()
    assert fp8_utils.relaid_bf16_moe_layers(m) == []


# ---------------------------------------------------------------------------
# The delta checkpoint engine
# ---------------------------------------------------------------------------


def _rollout(override=None):
    quantization = types.SimpleNamespace(
        enabled=override is not None, build_quantization_config=lambda path: dict(override or {})
    )
    return types.SimpleNamespace(quantization=quantization)


def _model(native=None, text_native=None):
    hf_config = types.SimpleNamespace(quantization_config=native)
    if text_native is not None:
        hf_config.text_config = types.SimpleNamespace(quantization_config=text_native)
    return types.SimpleNamespace(hf_config=hf_config, local_path="/unused")


_NATIVE_FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


@pytest.mark.parametrize(
    "rollout, model",
    [
        (_rollout({"quant_method": "compressed-tensors"}), _model()),  # rollout.quantization
        (_rollout(), _model(native=_NATIVE_FP8)),  # the checkpoint's own config, not on the refit loader
        (_rollout(), _model(text_native=_NATIVE_FP8)),  # under text_config
    ],
)
def test_the_delta_engine_is_refused_for_any_quantized_rollout(rollout, model):
    with pytest.raises(ValueError, match="delta checkpoint engine"):
        quant_utils.refuse_delta_sync_if_quantized(rollout, model)


def test_the_delta_engine_serves_a_bf16_rollout():
    quant_utils.refuse_delta_sync_if_quantized(_rollout(), _model())
