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
"""Audit that the training side and the rollout side quantize the same layers.

Why. Matched train/rollout quantization presumes both sides quantize the same
layers, but each side decides on its own: training implicitly (a layer runs FP8
GEMMs iff its TE linear modules run inside ``fp8_autocast``: the decoder layers,
minus ``first_last_layers_bf16``; never the embedding, the mcore output layer or
the torch-implemented MoE router), rollout by name (the trainer-side exporter
from ``rollout.quantization``, the engine from the configuration it was started
with). On B200, vLLM 0.24's ModelOpt MXFP8 config matched the router exclusion
``mlp.gate`` as a prefix of the dense ``mlp.gate_up_proj``: 56 of 311 parameters
of Qwen3-1.7B were served in BF16 while training ran them in FP8, and nothing
reported it.

What. Three sources:

- training: which decoder layers hold a TE FP8 weight workspace
  (``module._fp8_workspaces``), per mcore ``TransformerLayer``;
- export: what the sync actually sends, read from the trainer's weight stream
  after the rollout weight quantizer (``dispatch_rollout_weight_quantizer``): a
  quantized weight arrives in FP8 (or packed FP4, as ``weight_packed``), every
  other weight in high precision;
- engine: on vLLM, the quant method of the module that receives each weight, read
  back from the engine after the sync (``vllm_quant_methods``). SGLang has no
  return channel; its MXFP8 refit loader compares each incoming weight's dtype
  with the engine parameter that receives it and refuses a mismatch before
  anything is written.

Training vs export skips the modules the exporter keeps in BF16 by design (its
``DEFAULT_IGNORE``: ``lm_head``, embeddings, vision towers, MTP layers, routers,
the shared-expert gate) and judges the GEMM projections of each decoder layer
(attention, MLP, experts, linear-attention / SSM in and out projections) against
that layer's FP8 trace. A decoder-layer weight of any other kind, such as a
convolution, is not judged, since the layer-level trace says nothing about it;
the audit reports how many it left out. Export vs engine reports every
disagreement: both sides read ``rollout.quantization``, so one means the engine
matched a layer differently from the exporter.

Each comparison runs once. Training vs export waits for the first weight sync that
happens after an FP8 forward (the first sync, before any step, has no workspaces
yet). Export vs engine on vLLM needs only the export and the engine, so it runs at
the first weight sync, before the first rollout. Controlled by
``VERL_QUANT_LAYER_AUDIT``: ``warn`` (default) logs the disagreements, ``raise``
raises ``RuntimeError``, ``0`` disables.

Two things can remove the training-side signal, and the audit says so instead of
staying silent. verl drops the workspaces when it offloads the training model
(``offload_megatron_model_to_cpu``, at the end of every ``train_mode()`` when
``param_offload`` is on), so the clearing helper first marks every module that
held one (``remember_fp8_workspaces``) and the audit accepts the marker. And TE
only caches the FP8 weight when mcore passes ``is_first_microbatch``, which
``disable_parameter_transpose_cache=True`` turns off; with that flag there is
nothing to read, and the audit warns once that it cannot run.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Optional

import torch


def _configure_audit_logger() -> logging.Logger:
    """Route the audit's own output to stderr, past whatever the root logger was set to.

    The audit's evidence lines are INFO/WARNING. Inside the trainer worker process a vLLM transitive
    dependency (``model_hosting_container_standards.logging_config.configure_root_logger``, pulled in
    for SageMaker) raises the root logger and every root handler to ERROR at import time; anything that
    reaches this logger and then propagates to the root handlers is silently dropped - which is why the
    audit was invisible on the first B200 runs even though it was executing. A dedicated stderr handler
    on this logger with ``propagate=False`` carries the audit's INFO/WARNING lines out on their own,
    independent of the root configuration.
    """
    log = logging.getLogger(__name__)
    log.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO").upper())
    log.propagate = False
    handler_name = "verl.quant_audit.stderr"
    if not any(getattr(h, "name", None) == handler_name for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(handler_name)
        handler.setLevel(logging.NOTSET)  # let the logger's own level decide
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [quant-audit pid=%(process)d] %(message)s"))
        log.addHandler(handler)
    return log


logger = _configure_audit_logger()

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Routers (and mcore's shared-expert gate) are plain torch linears on the training side, never TE,
# regardless of how the HF checkpoint names them.
_ROUTER_RE = re.compile(r"(?:^|\.)(gate|router|shared_expert_gate)\.weight$")
# Non-linear leaves inside a decoder layer that never see a TE GEMM.
_NON_LINEAR_RE = re.compile(r"(norm|embed|rotary|bias$)", re.IGNORECASE)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# compressed-tensors' name for a packed FP4 weight; the module's ``weight`` in every other format.
_PACKED_SUFFIX = ".weight_packed"
_NO_QUANT_METHOD = "unquantized (no quant_method)"
_MAX_LISTED = 40


def audit_mode() -> str:
    return os.environ.get("VERL_QUANT_LAYER_AUDIT", "warn").strip().lower()


def trace(where: str, **state) -> None:
    """Log one checkpoint of the audit's own execution, at WARNING, when tracing is on.

    A guard that produces nothing is impossible to diagnose from the outside: telling "never
    installed" from "installed but never reached" from "reached but found nothing" otherwise needs
    instrumented runs. ``VERL_QUANT_LAYER_AUDIT_TRACE=1`` makes each of those states visible in one
    run, at a level no logging configuration drops.
    """
    if os.environ.get("VERL_QUANT_LAYER_AUDIT_TRACE", "0") != "1":
        return
    detail = " ".join(f"{k}={v}" for k, v in state.items())
    logger.warning("quantized-layer audit trace: %s%s", where, f" | {detail}" if detail else "")


# ---------------------------------------------------------------------------
# Training side: which decoder layers ran FP8 GEMMs
# ---------------------------------------------------------------------------


@dataclass
class TrainFp8Report:
    """Which decoder layers ran FP8 GEMMs on this rank, read from TE's workspaces."""

    seen_layers: set[int] = field(default_factory=set)
    quantized_layers: set[int] = field(default_factory=set)
    outside_layers_quantized: list[str] = field(default_factory=list)
    # mcore's TE wrappers carry config.disable_parameter_transpose_cache; True means TE never caches
    # the FP8 weight (is_first_microbatch=None), so no workspace can ever appear.
    transpose_cache_disabled: bool = False

    @property
    def active(self) -> bool:
        return bool(self.quantized_layers) or bool(self.outside_layers_quantized)


def fp8_recipe_of(engine_config) -> Optional[str]:
    """The Transformer Engine FP8 recipe a Megatron engine trains with, or None for BF16 GEMMs.

    Read from ``override_transformer_config``, the keys the Megatron provider receives: ``fp8``
    turns TE's FP8 GEMMs on and ``fp8_recipe`` picks the scaling recipe (Megatron's default is
    ``delayed``). Engines without that config return None.
    """
    overrides = getattr(engine_config, "override_transformer_config", None)
    if not overrides or not overrides.get("fp8"):
        return None
    return str(overrides.get("fp8_recipe") or "delayed").lower()


# Set on a TE module whose FP8 weight workspace verl has dropped (param offload); survives the offload.
FP8_WORKSPACE_SEEN_ATTR = "_verl_fp8_workspace_seen"


def remember_fp8_workspaces(module: torch.nn.Module) -> bool:
    """Mark ``module`` as having held an FP8 weight workspace. Call before clearing ``_fp8_workspaces``."""
    ws = getattr(module, "_fp8_workspaces", None)
    if isinstance(ws, dict) and len(ws) > 0:
        setattr(module, FP8_WORKSPACE_SEEN_ATTR, True)
        return True
    return False


def _has_fp8_workspace(module: torch.nn.Module) -> bool:
    ws = getattr(module, "_fp8_workspaces", None)
    return (isinstance(ws, dict) and len(ws) > 0) or bool(getattr(module, FP8_WORKSPACE_SEEN_ATTR, False))


def _is_transformer_layer(module: torch.nn.Module) -> bool:
    return isinstance(getattr(module, "layer_number", None), int) and "TransformerLayer" in type(module).__name__


def collect_train_fp8_report(modules: Iterable[torch.nn.Module] | torch.nn.Module) -> TrainFp8Report:
    """Walk Megatron model chunks and record, per global decoder layer, whether any TE module quantized."""
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    report = TrainFp8Report()
    for chunk in modules:
        in_layer: set[int] = set()
        for _, layer in chunk.named_modules():
            if not _is_transformer_layer(layer):
                continue
            idx = layer.layer_number - 1  # mcore layer_number is 1-based and global across PP
            report.seen_layers.add(idx)
            for sub in layer.modules():
                in_layer.add(id(sub))
                if getattr(sub, "disable_parameter_transpose_cache", False):
                    report.transpose_cache_disabled = True
                if _has_fp8_workspace(sub):
                    report.quantized_layers.add(idx)
        for name, sub in chunk.named_modules():
            if id(sub) not in in_layer and _has_fp8_workspace(sub):
                report.outside_layers_quantized.append(name)
    return report


# transformers >= 5 saves MoE experts fused and without a ``.weight`` suffix
# (``mlp.experts.gate_up_proj`` [E, 2I, H], ``mlp.experts.down_proj`` [E, H, I]).
_FUSED_EXPERT_RE = re.compile(r"\.experts\.(gate_up_proj|down_proj)$")


def is_linear_weight_name(name: str) -> bool:
    return name.endswith(".weight") or bool(_FUSED_EXPERT_RE.search(name))


# HF leaf modules of the projections a decoder layer runs as TE GEMMs: attention (including MLA), MLP,
# experts (Mixtral's w1/w2/w3 too) and the in / out projections of linear-attention and SSM mixers.
_GEMM_WEIGHT_RE = re.compile(
    r"(?:^|\.)(?:q_proj|k_proj|v_proj|o_proj|qkv_proj|query_key_value|q_a_proj|q_b_proj|kv_a_proj_with_mqa|"
    r"kv_b_proj|gate_proj|up_proj|down_proj|gate_up_proj|in_proj\w*|out_proj|fc1|fc2|w1|w2|w3)\.weight$"
)


def is_gemm_weight_name(name: str) -> bool:
    return bool(_GEMM_WEIGHT_RE.search(name) or _FUSED_EXPERT_RE.search(name))


def _of_unjudged_kind(name: str) -> bool:
    """A decoder-layer weight that is neither a known GEMM projection nor a known non-GEMM leaf."""
    return (
        is_linear_weight_name(name)
        and _LAYER_RE.search(name) is not None
        and not (_ROUTER_RE.search(name) or _NON_LINEAR_RE.search(name))
        and not is_gemm_weight_name(name)
    )


def expected_train_quantized(name: str, report: TrainFp8Report) -> Optional[bool]:
    """Training-side expectation for one HF weight name; ``None`` when this rank cannot judge."""
    if not is_linear_weight_name(name):
        return False
    m = _LAYER_RE.search(name)
    if m is None:
        # embeddings, lm_head, vision towers, ...: mcore quantizes nothing outside the decoder stack
        return None if report.outside_layers_quantized else False
    idx = int(m.group(1))
    if idx not in report.seen_layers:
        return None  # another pipeline stage owns this layer
    if _ROUTER_RE.search(name) or _NON_LINEAR_RE.search(name):
        return False
    if not is_gemm_weight_name(name):
        return None  # e.g. a convolution: the layer's FP8 trace does not say how training ran it
    return idx in report.quantized_layers


# ---------------------------------------------------------------------------
# Export side: what the weight sync sends
# ---------------------------------------------------------------------------


def exported_weight(name: str, tensor: torch.Tensor) -> Optional[tuple[str, bool, str]]:
    """``(weight name, quantized, format)`` for one entry of the exported stream; ``None`` for scales and the rest."""
    if name.endswith(_PACKED_SUFFIX):
        return name[: -len("_packed")], True, "packed FP4"
    if not is_linear_weight_name(name):
        return None
    dtype = getattr(tensor, "dtype", None)
    return name, dtype in _FP8_DTYPES, str(dtype).removeprefix("torch.")


def kept_high_precision_by_design(name: str) -> bool:
    """Whether the exporter keeps this weight's module in BF16 whatever the preset (its ``DEFAULT_IGNORE``)."""
    from verl.utils.qat.compressed_tensors import DEFAULT_IGNORE, _matches

    module = name[: -len(".weight")] if name.endswith(".weight") else name
    return _matches(module, DEFAULT_IGNORE)


def unjudged_weights(exported: Mapping[str, tuple[bool, str]]) -> list[str]:
    """Exported decoder-layer weights of a kind the training-vs-export comparison does not judge."""
    return [name for name in exported if _of_unjudged_kind(name) and not kept_high_precision_by_design(name)]


def _unjudged_note(unjudged: list[str]) -> str:
    if not unjudged:
        return ""
    shown = ", ".join(unjudged[:3]) + (", ..." if len(unjudged) > 3 else "")
    return f"; {len(unjudged)} decoder-layer weights of a kind the audit does not judge, e.g. {shown}"


def audit_train_vs_export(report: TrainFp8Report, exported: Mapping[str, tuple[bool, str]]) -> tuple[list[str], int]:
    """One line per exported weight on which training and the export disagree, and the by-design skip count."""
    problems = []
    by_design = 0
    for name, (quantized, fmt) in exported.items():
        if kept_high_precision_by_design(name):
            by_design += 1
            continue
        expected = expected_train_quantized(name, report)
        if expected is None:
            continue
        if quantized and not expected:
            problems.append(f"{name}: the export sends it in {fmt}, training ran it in high precision")
        elif expected and not quantized:
            problems.append(f"{name}: training ran it in FP8, the export sends it in {fmt}")
    return problems, by_design


# ---------------------------------------------------------------------------
# Engine side (vLLM): the quant method of the module that receives each weight
# ---------------------------------------------------------------------------


def engine_method_quantizes(method: str) -> bool:
    return not method.lower().startswith("unquantized")


def _vllm_receiving_module(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    """The vLLM module that receives HF weight ``name``, or ``None`` when the name does not resolve.

    Applies the model's ``hf_to_vllm_mapper``, then its ``packed_modules_mapping`` for the last
    module name (``q_proj`` -> ``qkv_proj``), and stops at a fused-MoE block, whose remaining
    per-expert parts (``experts.3.gate_proj``) its own weight loader resolves.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None:
        mapped = mapper.apply_list([name])
        if not mapped:
            return None
        name = mapped[0]
    parts = name.split(".")
    path = parts[:-1] if parts[-1] in ("weight", "weight_packed") else parts
    packed = getattr(model, "packed_modules_mapping", None) or {}
    owner = {sub: fused for fused, subs in packed.items() for sub in subs}
    module = model
    for i, part in enumerate(path):
        if hasattr(module, "routed_experts") or hasattr(module, "w13_weight"):
            break
        child = getattr(module, part, None)
        if child is None and i == len(path) - 1 and part in owner:
            child = getattr(module, owner[part], None)
        if not isinstance(child, torch.nn.Module):
            return None
        module = child
    for holder in ("routed_experts", "base_layer"):  # the MoE weight holder; the layer under a LoRA wrapper
        inner = getattr(module, holder, None)
        if isinstance(inner, torch.nn.Module):
            module = inner
    return module


def vllm_quant_methods(model: torch.nn.Module, names: Iterable[str]) -> list[Optional[str]]:
    """For each HF weight name, the quant method of the vLLM module that receives it (``None``: unresolved).

    Runs inside the vLLM worker. A compressed-tensors linear layer reports its scheme as well, e.g.
    ``CompressedTensorsLinearMethod (CompressedTensorsW8A8Mxfp8)``; a layer the config skips reports
    ``UnquantizedLinearMethod``. Layers another pipeline stage owns are unresolved on this rank.
    """
    methods: list[Optional[str]] = []
    for name in names:
        module = _vllm_receiving_module(model, name)
        if module is None or type(module).__name__ == "PPMissingLayer":
            methods.append(None)
            continue
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            methods.append(_NO_QUANT_METHOD)
            continue
        method = type(quant_method).__name__
        scheme = getattr(module, "scheme", None)
        methods.append(f"{method} ({type(scheme).__name__})" if scheme is not None else method)
    return methods


def merge_worker_methods(names: list[str], per_worker: Iterable) -> Optional[dict[str, Optional[str]]]:
    """Combine ``vllm_quant_methods`` answers of several workers: the first worker that resolves a name wins."""
    merged: dict[str, Optional[str]] = dict.fromkeys(names)
    answered = False
    for methods in per_worker or ():
        if not isinstance(methods, list) or len(methods) != len(names):
            continue
        answered = True
        for name, method in zip(names, methods, strict=True):
            if merged[name] is None:
                merged[name] = method
    return merged if answered else None


def audit_export_vs_engine(
    exported: Mapping[str, tuple[bool, str]], methods: Mapping[str, Optional[str]]
) -> tuple[list[str], int]:
    """One line per exported weight whose format disagrees with the engine module that receives it."""
    problems = []
    checked = 0
    for name, (quantized, fmt) in exported.items():
        method = methods.get(name)
        if method is None:
            continue
        checked += 1
        if engine_method_quantizes(method) != quantized:
            problems.append(f"{name}: the export sends it in {fmt}, the engine module that receives it runs {method}")
    return problems, checked


def _listing(problems: list[str]) -> str:
    return (
        "\n  - "
        + "\n  - ".join(problems[:_MAX_LISTED])
        + (f"\n  - ... and {len(problems) - _MAX_LISTED} more" if len(problems) > _MAX_LISTED else "")
    )


# ---------------------------------------------------------------------------
# Worker integration
# ---------------------------------------------------------------------------


class QuantLayerAuditor:
    """Records one weight sync's exported weights and compares them with training and, on vLLM, the engine.

    ``record`` wraps the training engine's weight export after the rollout weight quantizer, the one
    point every sync route passes through (the colocated worker, the checkpoint engine, server
    replicas), and runs the training-vs-export comparison when the stream ends. On vLLM the worker
    then asks the engine for the quant methods of the receiving modules and calls
    ``recheck_against_engine``.
    """

    def __init__(self, mode: str, engine: str = "", enabled: bool = True):
        self.mode = mode
        self.engine = engine
        self.enabled = enabled and mode not in ("0", "off", "false")
        self.done = False
        # HF weight name -> (sent quantized, format), from the last recorded sync
        self.exported: dict[str, tuple[bool, str]] = {}
        self.syncs_seen = 0  # distinct weight syncs the auditor wrapped; the give-up threshold counts these
        self._explained = False  # one-shot "why this guard produced nothing" warning
        self._verdict: Optional[TrainFp8Report] = None  # set by a training-vs-export verdict
        self._rechecked = False

    @classmethod
    def create(cls, *, rollout_quantized: bool, trainer_fp8_recipe: Optional[str], rollout_name: str):
        """Arm the audit for a TE FP8 trainer whose rollout is quantized; otherwise return a disabled auditor."""
        mode = audit_mode()
        if not rollout_quantized or trainer_fp8_recipe is None:
            logger.info(
                "quantized-layer audit: not armed (rollout quantized=%s, training FP8 recipe=%s): no matched "
                "quantization to compare",
                rollout_quantized,
                trainer_fp8_recipe,
            )
            return cls(mode, rollout_name, enabled=False)
        logger.info(
            "quantized-layer audit: armed (engine=%s, training FP8 recipe=%s, mode=%s); training vs export runs at "
            "the first weight sync after a training step, export vs engine (vLLM) at the first weight sync",
            rollout_name,
            trainer_fp8_recipe,
            mode,
        )
        return cls(mode, rollout_name)

    def record(self, weights: Iterable[tuple[str, torch.Tensor]], modules=None) -> Iterator[tuple[str, torch.Tensor]]:
        """Pass the exported stream through, remembering each weight's format; compare when it ends.

        ``modules`` (the training model chunks) makes the comparison run as soon as the stream is
        exhausted, wherever the stream is consumed.
        """
        trace("record", enabled=self.enabled, done=self.done, with_modules=modules is not None)
        if not self.enabled or self.done:
            yield from weights
            return
        self.exported = {}
        self.syncs_seen += 1  # one wrapped weight stream == one sync
        for name, tensor in weights:
            entry = exported_weight(name, tensor)
            if entry is not None:
                self.exported[entry[0]] = entry[1:]
            yield name, tensor
        trace("record.stream_end", weights=len(self.exported), will_run=modules is not None)
        if modules is not None:
            self.run(modules)

    def _explain_silence(self, reason: str) -> None:
        """Warn once that the audit produced no verdict: a guard that silently does nothing is the failure
        this module exists to prevent, so the reason is reported at WARNING."""
        if self._explained:
            return
        self._explained = True
        logger.warning(
            "quantized-layer audit produced no verdict (%s). engine=%s weights=%d. The train/rollout layer sets "
            "are unverified for this run; VERL_QUANT_LAYER_AUDIT=0 silences the audit entirely.",
            reason,
            self.engine or "?",
            len(self.exported),
        )

    def _report(self, msg: str) -> None:
        if self.mode == "raise":
            raise RuntimeError(msg)
        logger.warning(msg)

    def run(self, modules) -> Optional[list[str]]:
        """After a sync: compare the training trace with the recorded export. ``None`` if not run."""
        trace("run", enabled=self.enabled, done=self.done, weights=len(self.exported))
        if self.done or not self.enabled:
            return None
        if not self.exported:
            self._explain_silence("the weight sync streamed no weights through the auditor")
            return None
        report = collect_train_fp8_report(modules)
        if not report.active:
            # Give up only after two distinct syncs left no FP8 trace, counted by record(), so a second
            # run() on the same sync cannot trip this before the first training step.
            if report.transpose_cache_disabled or self.syncs_seen >= 2:
                self.done = True
                cause = (
                    "disable_parameter_transpose_cache=True: TE caches the FP8 weight only when mcore passes "
                    "is_first_microbatch, which that flag turns off"
                    if report.transpose_cache_disabled
                    else "no TE module holds an FP8 weight workspace after a training step; check that "
                    "override_transformer_config.fp8 is set on the model that ran, and that nothing other "
                    "than offload_megatron_model_to_cpu clears module._fp8_workspaces"
                )
                self._explain_silence(
                    f"no training-side FP8 trace on this rank: 0 of {len(report.seen_layers)} decoder layers "
                    f"left an FP8 weight workspace. Cause: {cause}"
                )
                return None
            logger.info(
                "quantized-layer audit: no FP8 trace yet on this rank (%d decoder layers seen, %d weights recorded); "
                "expected before the first training step, will retry at the next sync",
                len(report.seen_layers),
                len(self.exported),
            )
            return None
        problems, by_design = audit_train_vs_export(report, self.exported)
        unjudged = unjudged_weights(self.exported)
        self.done = True
        self._verdict = report
        n_q, n_seen = len(report.quantized_layers), len(report.seen_layers)
        if not problems:
            logger.info(
                "quantized-layer audit: training and the exported weights agree (%d of %d decoder layers on this "
                "rank ran FP8 GEMMs; %d weights checked, %d kept in BF16 on the rollout by design%s)",
                n_q,
                n_seen,
                len(self.exported) - by_design - len(unjudged),
                by_design,
                _unjudged_note(unjudged),
            )
            return []
        if self.engine == "sglang":
            engine_note = (
                " On SGLang the MXFP8 refit loader compares the export with the engine's parameters when it loads."
            )
        elif self.engine == "vllm":
            engine_note = " On vLLM the export is compared with the engine's quant methods at the first weight sync."
        else:
            engine_note = ""
        self._report(
            f"quantized-layer audit: training and the exported weights disagree on {len(problems)} weight(s) "
            f"({n_q} of {n_seen} decoder layers on this rank ran FP8 GEMMs{_unjudged_note(unjudged)}):"
            + _listing(problems)
            + "\nThe trainer ran these layers in one precision and the rollout is served them in another. Fix with "
            "rollout.quantization.ignore (rollout) or first_last_layers_bf16 / model wiring (training)."
            + engine_note
            + " VERL_QUANT_LAYER_AUDIT=raise turns this into an error, =0 silences it."
        )
        return problems

    def wants_engine_recheck(self) -> bool:
        """True once a sync has recorded the export and the engine has not been asked yet.

        Export vs engine needs no training-side trace, so it runs at the first weight sync, before the first
        rollout, instead of waiting for the training-vs-export verdict.
        """
        return self.enabled and bool(self.exported) and not self._rechecked

    def weight_names(self) -> list[str]:
        return list(self.exported)

    def recheck_against_engine(self, methods: Optional[Mapping[str, Optional[str]]]) -> Optional[list[str]]:
        """Compare the recorded export with the quant methods of the engine modules that receive it."""
        if self._rechecked:
            return None
        self._rechecked = True  # asked once; a rank whose rollout cannot be asked has no answer to wait for
        if not methods:
            return None
        problems, checked = audit_export_vs_engine(self.exported, methods)
        if not problems:
            logger.info(
                "quantized-layer audit: the exported weights match the %s engine's quant methods (%d weights "
                "checked, %d not resolved onto an engine module)",
                self.engine or "rollout",
                checked,
                len(self.exported) - checked,
            )
            return []
        self._report(
            f"quantized-layer audit: the exported weights and the {self.engine or 'rollout'} engine disagree on "
            f"{len(problems)} weight(s):"
            + _listing(problems)
            + "\nThe exporter and the engine both read rollout.quantization, so the engine matched these modules "
            "differently from the exporter; each such weight is loaded into a parameter of the other format. Check "
            "rollout.quantization.ignore and the engine version against docs/low_precision/mxfp8.md."
        )
        return problems
