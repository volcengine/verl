#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
解析当前目录（或指定通配符）的训练日志，抽取每 step 的 timing_s/* 与 perf/throughput，
并绘制：
 1) throughput_comparison.png （折线图，按配置对比 throughput）
 2) task_timing_breakdown.png （堆叠柱状图，分解 gen/update_actor/update_critic/values/adv/other 平均耗时）
 3) reward_loss_curves.png （折线图，记录 reward 与 loss 的变化）

解析规则：
- 使用正则逐行提取：
    - step：优先匹配 training/global_step（= 或 : 均可），否则在首次遇到 timing_s/step 或 perf/throughput 时自增步数。
    - 指标：匹配 key=value（key 允许包含 /._-），仅收集 timing_s/*、perf/throughput、包含 reward 或 loss 的指标。
    - throughput 的兜底：行内若出现 "examples/s" 或 "tokens/s"，会映射为 perf/throughput。
- 解析覆盖 1..116 步：若某日志不包含完整 116 个 step，则在对比图中忽略该日志并打印提示。
- 标签提取：
    - 优先从日志的 overrides 片段中解析 DP/TP/ZeRO/Mem（gpu_memory_utilization）。
    - 若无法解析，则从文件名尝试匹配 dp(\d+)_tp(\d+)_zero(\d+)_mem([0-9.]+)。
    - 最终回退为文件名。

输出：
- result/<stem>_parsed.csv：每个日志对应的宽表（step 为索引，列为 metric）。
- result/throughput_comparison.png
- result/task_timing_breakdown.png
- result/reward_loss_curves.png
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


STEP_TARGET = 116


@dataclass
class RunData:
    path: Path
    label: str
    df: pd.DataFrame  # index: step (int), columns: metrics
    complete: bool
    actor_zero: str | None = None


KV_RE = re.compile(r"(?P<k>[A-Za-z0-9_./-]+)\s*(?:=|:)\s*(?P<v>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
STEP_RE = re.compile(r"(?:^|\s)(?:training/global_step|global_step)\s*(?:=|:)\s*(\d+)")
EX_S_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:examples/s|tokens/s)")


def parse_label_from_overrides(lines: List[str]) -> Dict[str, str]:
    label: Dict[str, str] = {}
    for line in lines:
        if "overrides:" in line or "with overrides:" in line:
            # 提取 key=value
            for k, v in re.findall(r"([A-Za-z0-9_./-]+)=([^,'\]]+)", line):
                k = k.strip()
                v = v.strip("'\"")
                if k.endswith("actor_rollout_ref.actor.zero_stage"):
                    label["zero"] = v
                elif k.endswith("actor_rollout_ref.rollout.gpu_memory_utilization"):
                    label["mem"] = v
                elif k.endswith("actor_rollout_ref.rollout.tensor_model_parallel_size"):
                    label["tp"] = v
                elif k.endswith("trainer.n_gpus_per_node"):
                    label["ngpu"] = v
                elif k.endswith("trainer.nnodes"):
                    label["nnodes"] = v
    # 估算 DP
    if "tp" in label and "ngpu" in label:
        try:
            nnodes = int(label.get("nnodes", "1"))
            world = int(label["ngpu"]) * nnodes
            dp = max(1, world // int(label["tp"]))
            label["dp"] = str(dp)
        except Exception:
            pass
    return label


def derive_label(path: Path, lines: List[str]) -> Tuple[str, str | None]:
    # 先从 overrides 里解析
    kv = parse_label_from_overrides(lines)
    if kv:
        parts = []
        if "dp" in kv:
            parts.append(f"DP={kv['dp']}")
        if "tp" in kv:
            parts.append(f"TP={kv['tp']}")
        if "zero" in kv:
            parts.append(f"Z={kv['zero']}")
        if "mem" in kv:
            parts.append(f"Mem={kv['mem']}")
        if parts:
            return ", ".join(parts), kv.get("zero")

    # 再尝试从文件名提取
    m = re.search(r"dp(\d+)_tp(\d+)_zero(\d+)_mem([0-9.]+)", path.stem)
    if m:
        return f"DP={m.group(1)}, TP={m.group(2)}, Z={m.group(3)}, Mem={m.group(4)}", m.group(3)
    # 退化为文件名
    # 尝试仅提取 zero
    zm = re.search(r"zero(\d+)", path.stem)
    z = zm.group(1) if zm else None
    return path.stem, z


def parse_log_to_df(path: Path) -> RunData:
    lines = path.read_text(errors="ignore").splitlines()
    label, zero = derive_label(path, lines)

    # 收集数据：
    # per-step dict: step -> {metric: value}
    per_step: Dict[int, Dict[str, float]] = {}
    current_step = None
    encounter_count = 0  # 当没有显式 step 时自增步数

    MEM_FULL_RE = re.compile(
        r"memory allocated \(GB\):\s*([0-9.]+),\s*memory reserved \(GB\):\s*([0-9.]+),\s*device memory used/total \(GB\):\s*([0-9.]+)/([0-9.]+)",
        re.IGNORECASE,
    )

    for raw in lines:
        line = raw.strip()
        # step
        sm = STEP_RE.search(line)
        if sm:
            current_step = int(sm.group(1))

        # throughput 兜底
        em = EX_S_RE.search(line)
        if em:
            v = float(em.group(1))
            st = current_step
            if st is None:
                encounter_count += 1
                st = encounter_count
            per_step.setdefault(st, {})["perf/throughput"] = v

        # 标准 key=value
        for m in KV_RE.finditer(line):
            k = m.group("k")
            v = float(m.group("v"))
            if (
                k.startswith("timing_s/")
                or k == "perf/throughput"
                or ("reward" in k.lower())
                or ("loss" in k.lower() and "lossy" not in k.lower())
                or ("max_memory_allocated_gb" in k.lower())
            ):
                st = current_step
                if st is None:
                    encounter_count += 1
                    st = encounter_count
                per_step.setdefault(st, {})[k] = v

        # 精确解析 GPU 内存日志
        mm = MEM_FULL_RE.search(line)
        if mm:
            alloc_gb = float(mm.group(1))
            reserv_gb = float(mm.group(2))
            used_gb = float(mm.group(3))
            total_gb = float(mm.group(4))
            st = current_step
            if st is None:
                encounter_count += 1
                st = encounter_count
            per_step.setdefault(st, {})["memory/allocated_gb"] = alloc_gb
            per_step.setdefault(st, {})["memory/reserved_gb"] = reserv_gb
            per_step.setdefault(st, {})["memory/used_gb"] = used_gb
            per_step.setdefault(st, {})["memory/total_gb"] = total_gb

    if not per_step:
        df = pd.DataFrame()
        return RunData(path=path, label=label, df=df, complete=False, actor_zero=zero)

    # 组装 DataFrame
    steps = sorted(per_step.keys())
    df = pd.DataFrame.from_dict(per_step, orient="index").sort_index()
    df.index.name = "step"

    # 检查完整性：必须覆盖 1..116
    complete = all(s in df.index for s in range(1, STEP_TARGET + 1))
    return RunData(path=path, label=label, df=df, complete=complete, actor_zero=zero)


def save_parsed_csv(run: RunData, out_dir: Path) -> None:
    if run.df.empty:
        return
    out = out_dir / f"{run.path.stem}_parsed.csv"
    run.df.to_csv(out, float_format="%.6f")


def plot_throughput(runs: List[RunData], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    plotted = False
    for r in runs:
        if r.df.empty or not r.complete:
            print(f"[skip] {r.path.name}: steps not complete (need 1..{STEP_TARGET}).")
            continue
        if "perf/throughput" not in r.df.columns:
            print(f"[skip] {r.path.name}: no perf/throughput found.")
            continue
        xs = list(range(1, STEP_TARGET + 1))
        ys = r.df.reindex(xs)["perf/throughput"].values
        ax.plot(xs, ys, label=r.label, linewidth=1.8)
        plotted = True
    ax.set_title("Throughput Comparison")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    out = out_dir / "throughput_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


CORE_TASKS = [
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_critic",
    "timing_s/values",
    "timing_s/adv",
]


def plot_timing_breakdown(runs: List[RunData], out_dir: Path) -> Path:
    labels = []
    stacks: Dict[str, List[float]] = {t: [] for t in CORE_TASKS + ["timing_s/other"]}
    totals: List[float] = []

    for r in runs:
        if r.df.empty or not r.complete:
            continue
        df = r.df.reindex(range(1, STEP_TARGET + 1))
        if "timing_s/step" not in df.columns:
            print(f"[skip] {r.path.name}: no timing_s/step found.")
            continue
        labels.append(r.label)
        totals.append(df["timing_s/step"].mean())
        core_sums = 0.0
        for t in CORE_TASKS:
            val = df[t].mean() if t in df.columns else 0.0
            stacks[t].append(val)
            core_sums += val
        other = max(0.0, df["timing_s/step"].mean() - core_sums)
        stacks["timing_s/other"].append(other)

    if not labels:
        # 也生成空图以提示
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No complete runs for timing breakdown", ha="center")
        out = out_dir / "task_timing_breakdown.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return out

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))

    bottom = np.zeros(len(labels))
    colors = {
        "timing_s/gen": "#66c2a5",
        "timing_s/update_actor": "#fc8d62",
        "timing_s/update_critic": "#8da0cb",
        "timing_s/values": "#e78ac3",
        "timing_s/adv": "#a6d854",
        "timing_s/other": "#ffd92f",
    }

    for t in CORE_TASKS + ["timing_s/other"]:
        vals = stacks[t]
        ax.bar(x, vals, bottom=bottom, color=colors.get(t, None), label=t.replace("timing_s/", ""))
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Average Time per Step (seconds)")
    ax.set_title("Average Step Time Breakdown")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    out = out_dir / "task_timing_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_reward_loss(runs: List[RunData], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    ax_r, ax_l = axes
    any_r = any_l = False
    steps = list(range(1, STEP_TARGET + 1))
    for r in runs:
        if r.df.empty or not r.complete:
            continue
        df = r.df.reindex(steps)
        # reward: 只取 mean
        reward_cols = [c for c in df.columns if "reward" in c.lower() and "mean" in c.lower()]
        if reward_cols:
            col = reward_cols[0]
            ax_r.plot(steps, df[col].values, label=r.label, linewidth=1.6)
            any_r = True
        # loss：优先 loss 或 actor/critic loss
        loss_cols = [c for c in df.columns if c.lower().startswith("loss") or "/loss" in c.lower()]
        if loss_cols:
            col = loss_cols[0]
            ax_l.plot(steps, df[col].values, label=r.label, linewidth=1.6)
            any_l = True
    ax_r.set_title("Reward vs Steps")
    ax_r.set_xlabel("Global Step")
    ax_r.set_ylabel("Reward")
    ax_r.grid(True, alpha=0.3)
    if any_r:
        ax_r.legend(loc="best")

    ax_l.set_title("Loss vs Steps")
    ax_l.set_xlabel("Global Step")
    ax_l.set_ylabel("Loss")
    ax_l.grid(True, alpha=0.3)
    if any_l:
        ax_l.legend(loc="best")

    out = out_dir / "reward_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_per_log_panels(r: RunData, out_dir: Path) -> Path | None:
    """为单个日志输出一个 2x2 面板：reward、loss、throughput、memory usage。
    仅处理完整日志（1..116 步）。
    """
    if r.df.empty or not r.complete:
        print(f"[skip per-log] {r.path.name}: steps not complete (need 1..{STEP_TARGET}).")
        return None

    steps = list(range(1, STEP_TARGET + 1))
    df = r.df.reindex(steps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax_reward, ax_loss, ax_tp, ax_mem = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Reward（仅 mean）
    reward_cols = [c for c in df.columns if "reward" in c.lower() and "mean" in c.lower()]
    plotted_r = False
    if reward_cols:
        c = reward_cols[0]
        ax_reward.plot(steps, df[c].values, label=c, linewidth=1.6)
        plotted_r = True
    ax_reward.set_title("Reward")
    ax_reward.set_xlabel("Global Step")
    ax_reward.grid(True, alpha=0.3)
    if plotted_r:
        ax_reward.legend(fontsize=8, loc="best")

    # Loss（更宽泛地匹配包含 loss 的列，排除 lossy/loss_scale）
    loss_cols = [
        c
        for c in df.columns
        if ("loss" in c.lower()) and ("lossy" not in c.lower()) and ("loss_scale" not in c.lower())
    ]
    plotted_l = False
    for c in loss_cols:
        ax_loss.plot(steps, df[c].values, label=c, linewidth=1.6)
        plotted_l = True
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Global Step")
    ax_loss.grid(True, alpha=0.3)
    if plotted_l:
        ax_loss.legend(fontsize=8, loc="best")

    # Throughput
    if "perf/throughput" in df.columns:
        ax_tp.plot(steps, df["perf/throughput"].values, label="perf/throughput", linewidth=1.8)
    ax_tp.set_title("Throughput (tokens/sec)")
    ax_tp.set_xlabel("Global Step")
    ax_tp.grid(True, alpha=0.3)
    ax_tp.legend(fontsize=8, loc="best")

    # Memory usage（优先 max_memory_allocated_gb -> used_gb -> allocated/reserved）
    mem_cols = [c for c in df.columns if c.lower().startswith("memory/") or "memory" in c.lower() or "gpu" in c.lower() or "max_memory_allocated_gb" in c.lower()]
    plotted_m = False
    preferred = []
    # 1) 明确优先 max_memory_allocated_gb
    mm_cols = [c for c in df.columns if "max_memory_allocated_gb" in c.lower()]
    if mm_cols:
        preferred = [mm_cols[0]]
    elif "memory/used_gb" in df.columns:
        preferred = ["memory/used_gb"]
    elif "memory/allocated_gb" in df.columns or "memory/reserved_gb" in df.columns:
        preferred = [c for c in ["memory/allocated_gb", "memory/reserved_gb"] if c in df.columns]
    elif mem_cols:
        preferred = mem_cols[:1]
    for c in preferred:
        ax_mem.plot(steps, df[c].values, label=c, linewidth=1.6)
        plotted_m = True
    ax_mem.set_title("Memory Usage")
    ax_mem.set_xlabel("Global Step")
    ax_mem.set_ylabel("GB (if available)")
    ax_mem.grid(True, alpha=0.3)
    if plotted_m:
        ax_mem.legend(fontsize=8, loc="best")

    # 标题仅显示配置标签（用户自行记录 ZeRO）
    fig.suptitle(f"{r.label}")

    out = out_dir / f"perlog_{r.path.stem}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze training logs and plot metrics.")
    parser.add_argument("--logs", nargs="*", default=["*.log"], help="日志通配符（默认 *.log）")
    parser.add_argument("--out", default="result", help="输出目录")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 搜索日志文件
    paths: List[Path] = []
    for patt in args.logs:
        for p in sorted(glob.glob(os.path.expanduser(patt))):
            if p.endswith(".log"):
                paths.append(Path(p))
    if not paths:
        print("No log files found.")
        return 1

    runs: List[RunData] = []
    for p in paths:
        run = parse_log_to_df(p)
        save_parsed_csv(run, out_dir)
        runs.append(run)

    # 绘图（仅使用完整 116 步的日志）
    _ = plot_throughput(runs, out_dir)
    _ = plot_timing_breakdown(runs, out_dir)
    _ = plot_reward_loss(runs, out_dir)

    # 为每个完整日志输出单独的 2x2 面板
    for r in runs:
        _ = plot_per_log_panels(r, out_dir)

    # 打印完整性报告
    for r in runs:
        status = "OK" if r.complete else "INCOMPLETE"
        print(f"{r.path.name}: {status} (steps={len(r.df) if not r.df.empty else 0}) label={r.label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
