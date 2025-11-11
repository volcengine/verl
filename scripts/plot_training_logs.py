#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 VeRL 训练/推理日志生成高可读性的曲线图。

功能特性
- 自动解析常见设置（Ray 初始化、总步数等），生成一个“Settings”信息面板。
- 自动提取日志中的各类指标（key=value、examples/s 等），按指标绘制多子图曲线。
- 支持多日志文件叠加与聚合，自动去除 ANSI 颜色码并容错非标准行。

使用示例
    python verl/scripts/plot_training_logs.py \
        --logs tp_2_gpu_*.log dp_2_gpu_*.log \
        --out outputs/plots \
        --smooth 0

注意
- 本脚本在“尽量不假设固定日志格式”的前提下做了通用解析；若你在自定义路径输出了额外指标，
  也能被 key=value 模式自动拾取。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

# 后端设为 Agg 以便服务器/无显示环境
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------
# 基础工具
# -----------------------------

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE_RE.sub("", s)


def safe_literal_eval(text: str) -> Any:
    """更鲁棒的字典/列表解析。
    - 先尝试 ast.literal_eval
    - 失败后退回 json.loads（做布尔/None 替换）
    - 最后返回原始字符串
    """
    t = strip_ansi(text).strip()
    try:
        return ast.literal_eval(t)
    except Exception:
        try:
            t2 = t.replace("True", "true").replace("False", "false").replace("None", "null")
            return json.loads(t2)
        except Exception:
            return t


def rolling_mean(xs: List[float], k: int) -> List[float]:
    if k <= 1:
        return xs
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in xs:
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


# -----------------------------
# 日志解析
# -----------------------------

METRIC_KV_RE = re.compile(
    # 匹配 形如 key=value 或 key: value 的片段；key 允许包含 / _ - .
    r"(?P<key>[A-Za-z0-9_./-]+)\s*(?:=|:)\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
EXAMPLES_PER_S_RE = re.compile(r"(?P<eps>\d+(?:\.\d+)?)\s+examples/s")
IT_PER_S_RE = re.compile(r"(?P<ips>\d+(?:\.\d+)?)\s*(?:it/s|it\s*/\s*s)")


@dataclass
class ParsedRun:
    settings: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Tuple[int, float]]] = field(default_factory=lambda: defaultdict(list))
    filename: str = ""


def parse_log(path: Path) -> ParsedRun:
    run = ParsedRun(filename=path.name)
    step_counter = 0

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw.rstrip("\n"))
            if not line:
                continue

            # 提取设置类信息
            if "ray init kwargs:" in line:
                part = line.split("ray init kwargs:", 1)[1].strip()
                run.settings["ray_init_kwargs"] = safe_literal_eval(part)
                continue

            m = re.search(r"Total training steps:\s*(\d+)", line)
            if m:
                run.settings["total_training_steps"] = int(m.group(1))

            # 提取 hydra overrides（包含 batch size、zero stage 等）
            if "overrides:" in line or "with overrides:" in line:
                # 行内可能嵌套了 Python 列表样式
                # 先找所有 key=value 片段
                for ov in re.findall(r"([A-Za-z0-9_./-]+)=([^,'\]]+)", line):
                    k, v = ov
                    # 只收集若干关键配置
                    if any(
                        k.endswith(suf)
                        for suf in [
                            "data.train_batch_size",
                            "actor_rollout_ref.actor.ppo_mini_batch_size",
                            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
                            "actor_rollout_ref.actor.gradient_accumulation_steps",
                            "actor_rollout_ref.actor.zero_stage",
                            "critic.ppo_mini_batch_size",
                            "critic.ppo_micro_batch_size_per_gpu",
                            "critic.gradient_accumulation_steps",
                            "trainer.total_training_steps",
                        ]
                    ):
                        run.settings[k] = v.strip("'\"")

            # 提取 examples/s、it/s
            m = EXAMPLES_PER_S_RE.search(line)
            if m:
                val = float(m.group("eps"))
                run.metrics["throughput/examples_per_s"].append((step_counter, val))
            m = IT_PER_S_RE.search(line)
            if m:
                val = float(m.group("ips"))
                run.metrics["throughput/it_per_s"].append((step_counter, val))

            # 通用 key=value 或 key: value 指标
            # 仅收集“像指标”的 key
            def _is_metric_key(name: str) -> bool:
                n = name.strip()
                if not n:
                    return False
                lower = n.lower()
                # 明确排除
                if lower in {"pid", "rank", "seqnum", "timeout(ms)", "optype", "guid", "warning", "info", "error"}:
                    return False
                if lower in {"step", "steps", "epoch", "iter", "iteration"}:
                    return False
                if ".py" in lower or ".so" in lower or ".pt" in lower:
                    return False
                if ":" in n or "/:" in n:
                    return False
                # 类别判定
                if "/" in n:
                    top = n.split("/", 1)[0].lower()
                    allowed_top = {
                        "rewards", "reward", "throughput", "timing", "actor", "critic", "metrics",
                        "train", "eval", "validation", "rl", "ppo", "sft", "rm",
                    }
                    return top in allowed_top
                simple_prefixes = [
                    "loss", "kl", "entropy", "learning_rate", "lr", "ppl", "accuracy", "reward",
                    "timing", "tokens", "latency", "speed", "throughput",
                    "prompt_length", "response_length",
                ]
                return any(lower.startswith(p) for p in simple_prefixes)

            for kv in METRIC_KV_RE.finditer(line):
                key, sval = kv.group("key"), kv.group("val")
                if not _is_metric_key(key):
                    continue
                try:
                    val = float(sval)
                except Exception:
                    continue
                if key.lower() in {"step", "steps", "epoch", "iter", "iteration"}:
                    step_counter = int(val)
                else:
                    run.metrics[key].append((step_counter, val))

            # 兜底解析：memory 数值（GB/GiB/MB/MiB）
            if re.search(r"memory|mem|gpu", line, flags=re.IGNORECASE):
                m_gb = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB)", line)
                m_mb = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(MB|MiB)", line)
                if m_gb:
                    val = float(m_gb.group(1))
                    run.metrics["memory/GB"].append((step_counter, val))
                elif m_mb:
                    val = float(m_mb.group(1)) / 1024.0
                    run.metrics["memory/GB"].append((step_counter, val))

            step_counter += 1

    return run


# -----------------------------
# 绘图
# -----------------------------

def _format_settings_text(settings: Dict[str, Any]) -> str:
    lines: List[str] = []
    if not settings:
        return "No settings parsed."
    if "total_training_steps" in settings:
        lines.append(f"total_training_steps: {settings['total_training_steps']}")
    if "ray_init_kwargs" in settings:
        raykw = settings["ray_init_kwargs"]
        try:
            # 展开几项常用字段
            num_cpus = raykw.get("num_cpus") or raykw.get("num_cpus_per_worker")
            num_gpus = raykw.get("num_gpus") or raykw.get("num_gpus_per_worker")
            env_vars = raykw.get("runtime_env", {}).get("env_vars", {})
        except Exception:
            num_cpus = num_gpus = None
            env_vars = {}

        lines.append(f"ray.num_cpus: {num_cpus}")
        lines.append(f"ray.num_gpus: {num_gpus}")
        if env_vars:
            # 仅展示几项关键 env
            keys = [
                "TOKENIZERS_PARALLELISM",
                "NCCL_DEBUG",
                "VLLM_LOGGING_LEVEL",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "NCCL_CUMEM_ENABLE",
            ]
            for k in keys:
                if k in env_vars:
                    lines.append(f"{k}: {env_vars[k]}")
    return "\n".join(lines)


def plot_runs(runs: List[ParsedRun], out_dir: Path, smooth: int = 1, max_subplots: int | None = None) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    # 归并所有指标名称
    all_metric_keys: List[str] = []
    seen = set()
    for run in runs:
        for k in run.metrics.keys():
            if k not in seen:
                all_metric_keys.append(k)
                seen.add(k)

    if not all_metric_keys:
        # 输出一个 settings 图
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        text = "\n\n".join(
            [f"[{r.filename}]\n" + _format_settings_text(r.settings) for r in runs]
        )
        ax.text(0.01, 0.99, text or "No metrics parsed.", va="top", ha="left", family="monospace")
        out = out_dir / "metrics_summary.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)
        return outputs

    # 对指标按分组排序，优先展示常见指标
    priority_prefix = [
        "rewards/", "reward/", "loss", "kl", "entropy", "learning_rate", "lr", "throughput/"
    ]
    def key_score(k: str) -> Tuple[int, str]:
        for i, p in enumerate(priority_prefix):
            if k.startswith(p):
                return (i, k)
        return (len(priority_prefix), k)

    metric_keys_sorted = sorted(all_metric_keys, key=key_score)

    # 子图布局
    K = len(metric_keys_sorted)
    if max_subplots is not None:
        K = min(K, max_subplots)
        metric_keys_sorted = metric_keys_sorted[:K]

    ncols = 2 if K >= 2 else 1
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False, sharex=False)

    # 逐指标绘制
    for idx, key in enumerate(metric_keys_sorted):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        for run in runs:
            pairs = run.metrics.get(key, [])
            if not pairs:
                continue
            xs = [p for (p, _) in pairs]
            ys = [v for (_, v) in pairs]
            ys_s = rolling_mean(ys, smooth)
            ax.plot(xs, ys_s, label=run.filename, linewidth=1.6)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # 填充空白子图
    for j in range(K, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # 左上角添加 Settings 文本框（只展示第一个 run 的设置，避免过长）
    if runs:
        text = _format_settings_text(runs[0].settings)
        fig.text(0.01, 0.99, text, va="top", ha="left", fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.8, edgecolor="#cccccc"))

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    out = out_dir / "metrics_grid.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    outputs.append(out)
    
    # 进一步：按“类别”划分到不同图片，每个类别一个图（多子图）
    # 类别定义：以 "/" 分隔的首段作为类别；没有分隔符的 key 自身为类别
    categories: Dict[str, List[str]] = defaultdict(list)
    for k in all_metric_keys:
        cat = k.split("/", 1)[0]
        categories[cat].append(k)

    for cat, cat_keys in sorted(categories.items()):
        # 限制每类的展示数量（可选）
        keys = sorted(cat_keys)
        Kc = len(keys)
        ncols = 2 if Kc >= 2 else 1
        nrows = math.ceil(Kc / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False)
        for idx, key in enumerate(keys):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            for run in runs:
                pairs = run.metrics.get(key, [])
                if not pairs:
                    continue
                xs = [p for (p, _) in pairs]
                ys = [v for (_, v) in pairs]
                ys_s = rolling_mean(ys, smooth)
                ax.plot(xs, ys_s, label=run.filename, linewidth=1.6)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

        # 关闭多余子图
        for j in range(Kc, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        # 左上角设置文本
        if runs:
            text = _format_settings_text(runs[0].settings)
            fig.text(0.01, 0.99, text, va="top", ha="left", fontsize=9, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.8, edgecolor="#cccccc"))

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        out_cat = out_dir / f"cat_{cat}.png"
        fig.savefig(out_cat, dpi=160)
        plt.close(fig)
        outputs.append(out_cat)

    return outputs


# -----------------------------
# CLI
# -----------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse VeRL logs and plot metrics.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="日志文件路径或通配符（例如 tp_*.log dp_*.log）",
    )
    parser.add_argument("--out", type=str, default="outputs/plots", help="输出图片目录")
    parser.add_argument("--smooth", type=int, default=1, help="滑动平均窗口大小（>=1，不平滑为1）")
    parser.add_argument(
        "--max-subplots", type=int, default=None, help="最多绘制的子图数量（默认全部）"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # 展开通配符
    import glob
    paths: List[Path] = []
    for patt in args.logs:
        matches = glob.glob(os.path.expanduser(patt))
        if not matches:
            # 尝试基于当前工作目录的 Path.glob
            matches = [str(p) for p in Path().glob(patt)]
        paths.extend(map(Path, sorted(matches)))

    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No log files found.")
        return 1

    runs = [parse_log(p) for p in paths]

    out_dir = Path(args.out)
    outputs = plot_runs(runs, out_dir=out_dir, smooth=max(1, int(args.smooth)), max_subplots=args.max_subplots)

    # 按日志分别出图（reward / loss / memory / throughput 四宫格）
    for run in runs:
        out = plot_per_log(run, out_dir, smooth=max(1, int(args.smooth)))
        print(f"Saved: {out}")
    return 0


def _collect_series(run: ParsedRun, key_predicate) -> List[Tuple[str, List[int], List[float]]]:
    series: List[Tuple[str, List[int], List[float]]] = []
    for k, pairs in run.metrics.items():
        if key_predicate(k.lower()):
            xs = [p for (p, _) in pairs]
            ys = [v for (_, v) in pairs]
            series.append((k, xs, ys))
    return series


def plot_per_log(run: ParsedRun, out_dir: Path, smooth: int = 1) -> Path:
    """为单个日志文件生成一张 2x2 图，包含 reward / loss / memory / throughput。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False, constrained_layout=True)
    ax_reward, ax_loss, ax_mem, ax_tp = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Reward
    reward_series = _collect_series(run, lambda k: ("reward" in k or "rewards" in k))
    for name, xs, ys in reward_series:
        ax_reward.plot(xs, rolling_mean(ys, smooth), label=name, linewidth=1.8)
    ax_reward.set_title("reward")
    ax_reward.set_xlabel("step")
    ax_reward.grid(True, alpha=0.3)
    if reward_series:
        ax_reward.legend(fontsize=8, loc="best")

    # Loss
    loss_series = _collect_series(run, lambda k: ("loss" in k and "throughput" not in k and "lossy" not in k))
    for name, xs, ys in loss_series:
        ax_loss.plot(xs, rolling_mean(ys, smooth), label=name, linewidth=1.8)
    ax_loss.set_title("loss")
    ax_loss.set_xlabel("step")
    ax_loss.grid(True, alpha=0.3)
    if loss_series:
        ax_loss.legend(fontsize=8, loc="best")

    # Memory (GB)
    mem_series = _collect_series(run, lambda k: (k.startswith("memory/") or "memory" in k))
    # 优先用 memory/GB
    preferred = [(n, xs, ys) for (n, xs, ys) in mem_series if n.lower().startswith("memory/gb")]
    plot_list = preferred if preferred else mem_series
    for name, xs, ys in plot_list:
        ax_mem.plot(xs, rolling_mean(ys, smooth), label=name, linewidth=1.8)
    ax_mem.set_title("memory (GB)")
    ax_mem.set_xlabel("step")
    ax_mem.grid(True, alpha=0.3)
    if plot_list:
        ax_mem.legend(fontsize=8, loc="best")

    # Throughput: examples/s, it/s
    tp_series = []
    for k in ["throughput/examples_per_s", "throughput/it_per_s"]:
        if k in run.metrics:
            pairs = run.metrics[k]
            xs = [p for (p, _) in pairs]
            ys = [v for (_, v) in pairs]
            tp_series.append((k, xs, ys))
    for name, xs, ys in tp_series:
        ax_tp.plot(xs, rolling_mean(ys, smooth), label=name, linewidth=1.8)
    ax_tp.set_title("throughput")
    ax_tp.set_xlabel("step")
    ax_tp.grid(True, alpha=0.3)
    if tp_series:
        ax_tp.legend(fontsize=8, loc="best")

    # 左上角汇总设置：batch size、zero stage 等
    info_lines: List[str] = []
    # 训练 batch
    for k in [
        "data.train_batch_size",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.actor.gradient_accumulation_steps",
        "critic.ppo_mini_batch_size",
        "critic.ppo_micro_batch_size_per_gpu",
        "critic.gradient_accumulation_steps",
    ]:
        if k in run.settings:
            info_lines.append(f"{k}: {run.settings[k]}")
    # zero stage
    if "actor_rollout_ref.actor.zero_stage" in run.settings:
        info_lines.append(f"actor.zero_stage: {run.settings['actor_rollout_ref.actor.zero_stage']}")
    if "ray_init_kwargs" in run.settings:
        rkw = run.settings["ray_init_kwargs"]
        try:
            num_cpus = rkw.get("num_cpus")
            num_gpus = rkw.get("num_gpus")
        except Exception:
            num_cpus = num_gpus = None
        info_lines.append(f"ray.cpus: {num_cpus}")
        info_lines.append(f"ray.gpus: {num_gpus}")
    if "total_training_steps" in run.settings:
        info_lines.append(f"total_steps: {run.settings['total_training_steps']}")

    fig.text(
        0.01,
        0.99,
        f"[{run.filename}]\n" + ("\n".join(info_lines) if info_lines else ""),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f7f7f7", alpha=0.8, edgecolor="#cccccc"),
    )

    out_path = out_dir / f"perlog_{Path(run.filename).stem}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    sys.exit(main())
