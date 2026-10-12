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
"""Manually benchmark shared score-centering normalization on an idle CUDA GPU.

Run from a checkout with the FSDP CUDA dependencies installed:

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python benchmarks/benchmark_score_centering.py \
        --n 4096 --v 151936 --k 128 --dtype bfloat16 --warmup 4 --rounds 6 \
        --output score_centering_benchmark.json

The synthetic fixture measures score/loss forward and dense logits-gradient backward.
It excludes model forward, optimizer, rollout, Ray, DP/SP and communication. Both routes
retain the same head-mass and alignment diagnostics; the candidate still needs a
full-vocabulary forward normalization for the detached trainer head-mass metric.
The baseline is the auto-correction helper plus separately computed sampled CE.
Timing uses identical inputs and ABBA order, after untimed numerical checks and warm-up.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def _source_metadata(root, sc):
    files = {
        "score_centering": Path(sc.__file__),
        "shared_scores": root / "verl/trainer/ppo/score_centering_scores.py",
        "sampled_ce": root / "verl/utils/torch_functional.py",
        "benchmark": Path(__file__),
    }
    metadata = {
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
    }
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
        metadata["git_head"] = revision.stdout.strip() if revision.returncode == 0 else None
        metadata["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except FileNotFoundError:
        metadata.update(git_head=None, git_dirty=None)
    return metadata


def _benchmark_case(args, dtype, aligned_fraction, sc, logprobs_from_logits):
    n, v, k = args.n, args.v, args.k
    rule = lambda ratio: ratio.clamp(max=2.0)
    logits = torch.randn(n, v, device="cuda", dtype=dtype)
    logits[:, :k] += 10.0
    logits.requires_grad_(True)
    ids = torch.arange(k, device="cuda").expand(n, k).contiguous()
    aligned = torch.arange(n, device="cuda") < int(n * aligned_fraction)
    q_head = torch.log_softmax(torch.randn(n, k, device="cuda") * 1.1, -1) + math.log(0.97)
    support_size = max(1, 3 * k // 4)
    support_q = q_head.masked_fill(torch.arange(k, device="cuda").unsqueeze(0) >= support_size, float("-inf"))
    support_q = support_q - support_q.logsumexp(-1, keepdim=True)
    q_head = torch.where(aligned.unsqueeze(-1), support_q, q_head)
    labels = torch.where(
        aligned,
        torch.zeros(n, device="cuda", dtype=torch.long),
        torch.full((n,), k + 7, device="cuda", dtype=torch.long),
    )
    q_sample = torch.where(aligned, q_head[:, 0], torch.full((n,), math.log(0.03 / (v - k)), device="cuda"))
    advantages = torch.randn(n, device="cuda")

    def baseline():
        head = sc.topk_log_probs_from_logits(logits, ids)
        correction, log_mass, q_mass, train_mass = sc.score_centering_auto_correction(head, q_head, rule, aligned)
        sample = logprobs_from_logits(logits, labels, inplace_backward=False) - log_mass
        tail_mass = (1 - q_mass).clamp(min=0)
        return {
            "sc_correction": correction,
            "log_probs": sample,
            "sc_train_head_mass": train_mass,
            "sc_sampler_head_mass": q_mass,
            "sc_log_support_mass": log_mass,
            "sc_support_aligned": aligned.float(),
            "sc_sampler_tail_mass": tail_mass,
            "sc_support_unknown": ((~aligned) & (tail_mass <= 1e-6)).float(),
        }

    def run(mode):
        logits.grad = None
        outputs = (
            baseline()
            if mode == "baseline"
            else sc.score_centering_shared_scores_from_logits(logits, ids, labels, q_head, aligned, rule)
        )
        lp = outputs["log_probs"]
        weights = rule((lp.detach() - q_sample).exp())
        loss = -(advantages * (weights * lp - outputs["sc_correction"])).mean()
        loss.backward()
        return loss.detach(), outputs

    baseline_loss, baseline_out = run("baseline")
    baseline_grad = logits.grad.clone()
    candidate_loss, candidate_out = run("candidate")
    # log_support_mass differs by contract: the candidate has already
    # normalized sampled scores, so it returns zero for that internal field.
    metric_names = (
        "sc_train_head_mass",
        "sc_sampler_head_mass",
        "sc_support_aligned",
        "sc_sampler_tail_mass",
        "sc_support_unknown",
    )
    parity = {
        "loss_abs_diff": (candidate_loss - baseline_loss).abs().item(),
        "grad_max_abs_diff": (logits.grad - baseline_grad).abs().max().item(),
        "sampled_score_max_abs_diff": (candidate_out["log_probs"] - baseline_out["log_probs"]).abs().max().item(),
        "metric_max_abs_diff": max(
            (candidate_out[name] - baseline_out[name]).abs().max().item() for name in metric_names
        ),
    }
    del baseline_out, candidate_out, baseline_grad, baseline_loss, candidate_loss
    for _ in range(args.warmup):
        run("baseline")
        run("candidate")
    torch.cuda.synchronize()
    samples = {"baseline": [], "candidate": []}
    for _ in range(args.rounds):
        for mode in ("baseline", "candidate", "candidate", "baseline"):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            run(mode)
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end))
    timing = {
        mode: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "count": len(values),
            "samples_ms": values,
        }
        for mode, values in samples.items()
    }
    peak = {}
    for mode in ("baseline", "candidate"):
        logits.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_bytes = torch.cuda.memory_allocated()
        run(mode)
        torch.cuda.synchronize()
        peak[mode] = {"initial_bytes": initial_bytes, "peak_bytes": torch.cuda.max_memory_allocated()}
    return {"aligned_fraction": aligned_fraction, "parity": parity, "timing": timing, "peak_allocated": peak}


def main():
    """Run the matched benchmark and emit its configuration, parity and cost as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=4096, help="Number of prediction positions.")
    parser.add_argument("--v", type=int, default=151936, help="Full local vocabulary size.")
    parser.add_argument("--k", type=int, default=128, help="Recorded head size.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=4, help="Untimed calls per route before ABBA timing.")
    parser.add_argument(
        "--rounds", type=int, default=6, help="ABBA rounds; each route gets two observations per round."
    )
    parser.add_argument("--seed", type=int, default=8196)
    parser.add_argument("--output", type=Path, help="Optional JSON output file; results also print to stdout.")
    args = parser.parse_args()
    if args.n <= 0 or args.k <= 0 or args.v <= args.k + 7:
        parser.error("Require n > 0, k > 0 and v > k + 7 for the sampled-outside-head fixture.")
    if args.warmup < 0 or args.rounds <= 0:
        parser.error("Require warmup >= 0 and rounds > 0.")
    if not torch.cuda.is_available():
        parser.error("This benchmark requires a CUDA GPU and the FSDP CUDA environment.")

    # Prefer this checkout even when another verl checkout is installed editable.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from verl.trainer.ppo import score_centering as sc
    from verl.trainer.ppo import score_centering_scores as sc_scores
    from verl.utils import torch_functional as verl_F

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    n, v, k = args.n, args.v, args.k
    dtype = getattr(torch, args.dtype)
    result = {
        **_source_metadata(root, sc),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "n": n,
        "v": v,
        "k": k,
        "seed": args.seed,
        "warmup_per_route": args.warmup,
        "abba_rounds": args.rounds,
        "baseline": "topk_log_probs_from_logits + score_centering_auto_correction + separate sampled CE",
        "candidate": "score_centering_shared_scores_from_logits",
        "weight_rule": "token TIS with upper threshold 2.0",
        "candidate_triton_available": sc_scores.triton is not None,
        "flash_attn_ce_available": verl_F.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
        "disable_flash_attn_ce": os.environ.get("VERL_DISABLE_FLASH_ATTN_CE", "0"),
        "peak_allocated_scope": "Includes inputs, outputs and logits gradients; excludes allocator reservation.",
        "scope": (
            "Synthetic logits, mixed SC, sampled score and dense logits gradient only; "
            "no model/rollout/Ray/optimizer/DP/SP/communication."
        ),
        "cases": [],
    }
    for aligned_fraction in (0.0, 0.5, 1.0):
        result["cases"].append(_benchmark_case(args, dtype, aligned_fraction, sc, verl_F.logprobs_from_logits))
        gc.collect()
        torch.cuda.empty_cache()

    serialized = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
