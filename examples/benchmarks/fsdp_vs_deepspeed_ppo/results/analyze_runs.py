#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""Analyze DeepSpeed/FSDP PPO logs and export consolidated metrics.

Key features:
- Robust ANSI stripping and tolerant numeric parsing.
- Config extraction covering DP/TP/PP/SP, batch sizes, precision/offload, ZeRO.
- Exports CSV/JSON/Markdown + multi-style plots.
- Per-setting 2x2 grid charts keeping the same layout as metrics_progress.png.
"""


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
NP_FLOAT_RE = re.compile(r"np\.float64\(([-+0-9.eE]+)\)")

METRIC_KEY_MAP = {
    "reward": "val-core/openai/gsm8k/reward/mean@1",
    "actor_entropy": "actor/entropy",
    "actor_pg_loss": "actor/pg_loss",
    "critic_vf_loss": "critic/vf_loss",
    "actor_grad_norm": "actor/grad_norm",
    "critic_grad_norm": "critic/grad_norm",
    "throughput": "perf/throughput",
}

CONFIG_KEY_PATTERNS = {
    # Topology
    "trainer.total_epochs": r"'total_epochs':\s*([^,}\n]+)",
    "trainer.n_gpus_per_node": r"'n_gpus_per_node':\s*([^,}\n]+)",
    "trainer.nnodes": r"'nnodes':\s*([^,}\n]+)",
    "tensor_model_parallel_size": r"'tensor_model_parallel_size':\s*([^,}\n]+)",
    "pipeline_model_parallel_size": r"'pipeline_model_parallel_size':\s*([^,}\n]+)",
    "data_parallel_size": r"'data_parallel_size':\s*([^,}\n]+)",
    "ulysses_sequence_parallel_size": r"'ulysses_sequence_parallel_size':\s*([^,}\n]+)",
    # Batch/accumulation
    "ppo_mini_batch_size": r"'ppo_mini_batch_size':\s*([^,}\n]+)",
    "ppo_micro_batch_size_per_gpu": r"'ppo_micro_batch_size_per_gpu':\s*([^,}\n]+)",
    "gradient_accumulation_steps": r"'gradient_accumulation_steps':\s*([^,}\n]+)",
    "train_batch_size": r"'train_batch_size':\s*([^,}\n]+)",
    "train_micro_batch_size_per_gpu": r"'train_micro_batch_size_per_gpu':\s*([^,}\n]+)",
    # Precision/offload/zero
    "model_dtype": r"'model_dtype':\s*([^,}\n]+)",
    "mixed_precision": r"'mixed_precision':\s*([^,}\n]+)",
    "optimizer_offload": r"'optimizer_offload':\s*([^,}\n]+)",
    "param_offload": r"'param_offload':\s*([^,}\n]+)",
    "zero_stage": r"'zero_stage':\s*([^,}\n]+)",
}


@dataclass
class StepMetrics:
    step: int
    raw: Dict[str, float]


@dataclass
class RunSummary:
    label: str
    log_path: Path
    steps: List[StepMetrics] = field(default_factory=list)
    final_validation: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Optional[int]] = field(default_factory=dict)

    @property
    def final_step(self) -> Optional[int]:
        if not self.steps:
            return None
        return max(s.step for s in self.steps)

    @property
    def final_metrics(self) -> Dict[str, Optional[float]]:
        if not self.steps:
            return {}
        latest = max(self.steps, key=lambda s: s.step)
        return latest.raw


def parse_value(value_text: str) -> Optional[float]:
    value_text = value_text.strip().strip("'\"")
    if not value_text:
        return None
    if value_text in {"None", "nan", "NaN"}:
        return None
    if value_text.lower() == "inf":
        return math.inf
    if value_text.lower() == "-inf":
        return -math.inf
    m = NP_FLOAT_RE.fullmatch(value_text)
    if m:
        return float(m.group(1))
    try:
        if "." in value_text or "e" in value_text.lower():
            return float(value_text)
        return float(int(value_text))
    except ValueError:
        return None


def clean_line(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "").strip()


def parse_log(path: Path, label: str) -> RunSummary:
    summary = RunSummary(label=label, log_path=path)
    config_search_space: List[str] = []
    collecting_final_block = False
    final_buffer: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = clean_line(raw_line)
            if not line:
                continue

            config_search_space.append(line)

            if "Final validation metrics" in line:
                collecting_final_block = True
                idx = line.index("Final validation metrics:") + len(
                    "Final validation metrics:"
                )
                payload = line[idx:].strip()
                if payload:
                    final_buffer.append(payload)
                if "}" in payload:
                    collecting_final_block = False
                    summary.final_validation = parse_final_validation(final_buffer)
                    final_buffer = []
                continue

            if collecting_final_block:
                final_buffer.append(line)
                if "}" in line:
                    collecting_final_block = False
                    summary.final_validation = parse_final_validation(final_buffer)
                    final_buffer = []
                continue

            if "step:" in line and "training/global_step" in line:
                step_metrics = parse_step_line(line)
                if step_metrics:
                    summary.steps.append(step_metrics)

    summary.config = extract_config_info(config_search_space)
    return summary


def parse_step_line(line: str) -> Optional[StepMetrics]:
    segments = [segment.strip() for segment in line.split(" - ") if ":" in segment]
    values: Dict[str, float] = {}
    step_id = None

    for segment in segments:
        key, value = segment.split(":", 1)
        key = key.strip()
        parsed_value = parse_value(value)
        if key == "step":
            step_id = int(parsed_value) if parsed_value is not None else None
        if key in METRIC_KEY_MAP.values():
            if parsed_value is not None:
                values[key] = parsed_value
        if key.startswith("training/global_step"):
            if parsed_value is not None:
                step_id = int(parsed_value)

    if step_id is None:
        return None

    normalized = {
        short_name: values.get(full_key) for short_name, full_key in METRIC_KEY_MAP.items()
    }

    return StepMetrics(step=step_id, raw=normalized)


def parse_final_validation(lines: Iterable[str]) -> Dict[str, float]:
    joined = " ".join(line.strip(" '\"") for line in lines)
    cleaned = NP_FLOAT_RE.sub(r"\1", joined)
    try:
        payload = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return {}
    result = {}
    for key, value in payload.items():
        parsed = parse_value(str(value))
        if parsed is not None:
            result[key] = parsed
    return result


def _parse_config_token(token: str) -> Optional[int | str | bool]:
    token = token.strip().strip(",")
    # Unquote if quoted
    if (token.startswith("'") and token.endswith("'")) or (
        token.startswith('"') and token.endswith('"')
    ):
        token = token[1:-1]
    # Booleans
    if token in ("True", "False"):
        return token == "True"
    # Numerics
    try:
        if re.fullmatch(r"[-+]?[0-9]+", token):
            return int(token)
        if re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", token):
            return float(token)
    except Exception:
        pass
    # Fallback to raw string snippet
    if token == "None":
        return None
    return token


def extract_config_info(lines: List[str]) -> Dict[str, Optional[int]]:
    snapshot = "\n".join(lines)
    config: Dict[str, Optional[int | str | bool]] = {}
    for logical_key, pattern in CONFIG_KEY_PATTERNS.items():
        matches = list(re.finditer(pattern, snapshot))
        if matches:
            raw = matches[-1].group(1)
            config[logical_key] = _parse_config_token(raw)
        else:
            config[logical_key] = None
    return add_derived_parallelism(config)


def add_derived_parallelism(config: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    result = dict(config)
    nodes = result.get("trainer.nnodes") or 0
    gpus_per_node = result.get("trainer.n_gpus_per_node") or 0
    total_gpus = nodes * gpus_per_node
    tp_size = result.get("tensor_model_parallel_size") or 1
    pp_size = result.get("pipeline_model_parallel_size") or 1
    if total_gpus > 0 and tp_size > 0 and pp_size > 0:
        derived = total_gpus // (tp_size * pp_size)
        if derived > 0:
            result["derived_data_parallel_size"] = derived
    return result


def ensure_run_files(run_map: Dict[str, Path]) -> Dict[str, Path]:
    missing = [label for label, path in run_map.items() if not path.exists()]
    if missing:
        lines = [f"{label}: {run_map[label]}" for label in missing]
        raise FileNotFoundError("Missing log files:\n" + "\n".join("  " + line for line in lines))
    return run_map


def build_default_run_map(results_dir: Path) -> Dict[str, Path]:
    return {
        "dp_2gpu_full": results_dir / "dp_2gpu_full.log",
        "tp_2gpu_full": results_dir / "tp_full.log",
        "dp_tp_ep1": results_dir / "dp_tp_full.log",
        "dp_tp_ep2": results_dir / "dp_tp_full_ep2.log",
    }


def merge_run_map(base_dir: Path, overrides: Optional[List[str]]) -> Dict[str, Path]:
    run_map = build_default_run_map(base_dir)
    if not overrides:
        return run_map
    for item in overrides:
        if ":" not in item:
            raise ValueError(f"Override must be LABEL:PATH, got {item}")
        label, relative_path = item.split(":", 1)
        run_map[label] = Path(relative_path).expanduser().resolve()
    return run_map


def summarize_runs(runs: Dict[str, Path]) -> List[RunSummary]:
    summaries = []
    for label, path in runs.items():
        summaries.append(parse_log(path, label))
    return summaries


def write_csv(summaries: List[RunSummary], output_path: Path) -> None:
    fieldnames = [
        "label",
        "log_path",
        "final_step",
        "derived_data_parallel_size",
        "nnodes",
        "n_gpus_per_node",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "ulysses_sequence_parallel_size",
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "zero_stage",
        "model_dtype",
        "mixed_precision",
        "param_offload",
        "optimizer_offload",
        "reward",
        "actor_entropy",
        "actor_pg_loss",
        "critic_vf_loss",
        "actor_grad_norm",
        "critic_grad_norm",
        "throughput",
        "final_validation_reward",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            final_metrics = summary.final_metrics
            writer.writerow(
                {
                    "label": summary.label,
                    "log_path": str(summary.log_path),
                    "final_step": summary.final_step,
                    "derived_data_parallel_size": summary.config.get("derived_data_parallel_size"),
                    "nnodes": summary.config.get("trainer.nnodes"),
                    "n_gpus_per_node": summary.config.get("trainer.n_gpus_per_node"),
                    "tensor_model_parallel_size": summary.config.get("tensor_model_parallel_size"),
                    "pipeline_model_parallel_size": summary.config.get("pipeline_model_parallel_size"),
                    "ulysses_sequence_parallel_size": summary.config.get("ulysses_sequence_parallel_size"),
                    "train_batch_size": summary.config.get("train_batch_size"),
                    "train_micro_batch_size_per_gpu": summary.config.get("train_micro_batch_size_per_gpu"),
                    "ppo_mini_batch_size": summary.config.get("ppo_mini_batch_size"),
                    "ppo_micro_batch_size_per_gpu": summary.config.get("ppo_micro_batch_size_per_gpu"),
                    "gradient_accumulation_steps": summary.config.get("gradient_accumulation_steps"),
                    "zero_stage": summary.config.get("zero_stage"),
                    "model_dtype": summary.config.get("model_dtype"),
                    "mixed_precision": summary.config.get("mixed_precision"),
                    "param_offload": summary.config.get("param_offload"),
                    "optimizer_offload": summary.config.get("optimizer_offload"),
                    "reward": safe_float(final_metrics.get("reward")),
                    "actor_entropy": safe_float(final_metrics.get("actor_entropy")),
                    "actor_pg_loss": safe_float(final_metrics.get("actor_pg_loss")),
                    "critic_vf_loss": safe_float(final_metrics.get("critic_vf_loss")),
                    "actor_grad_norm": safe_float(final_metrics.get("actor_grad_norm")),
                    "critic_grad_norm": safe_float(final_metrics.get("critic_grad_norm")),
                    "throughput": safe_float(final_metrics.get("throughput")),
                    "final_validation_reward": safe_float(
                        summary.final_validation.get("val-core/openai/gsm8k/reward/mean@1")
                    ),
                }
            )


def write_json(summaries: List[RunSummary], output_path: Path) -> None:
    payload = {}
    for summary in summaries:
        payload[summary.label] = {
            "log_path": str(summary.log_path),
            "final_step": summary.final_step,
            "final_metrics": summary.final_metrics,
            "final_validation": summary.final_validation,
            "config": summary.config,
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(summaries: List[RunSummary], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# DeepSpeed PPO Multi-run Summary\n")
    for summary in summaries:
        lines.append(f"## {summary.label}")
        lines.append(f"- Log: `{summary.log_path}`")
        lines.append(f"- Final step: {summary.final_step}")
        reward = format_metric(
            summary.final_metrics.get("reward")
            or summary.final_validation.get("val-core/openai/gsm8k/reward/mean@1")
        )
        lines.append(f"- Final reward: {reward}")
        lines.append("- Key metrics at final step:")
        for metric_key in (
            "actor_entropy",
            "actor_pg_loss",
            "critic_vf_loss",
            "actor_grad_norm",
            "critic_grad_norm",
            "throughput",
        ):
            metric_value = summary.final_metrics.get(metric_key)
            lines.append(f"  - {metric_key}: {format_metric(metric_value)}")
        lines.append("- 并行与批量设置：")
        for cfg_key in (
            "trainer.nnodes",
            "trainer.n_gpus_per_node",
            "derived_data_parallel_size",
            "data_parallel_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "ulysses_sequence_parallel_size",
            "ppo_mini_batch_size",
            "ppo_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "train_batch_size",
            "train_micro_batch_size_per_gpu",
        ):
            lines.append(f"  - {cfg_key}: {summary.config.get(cfg_key)}")

        lines.append("- 精度与内存：")
        for cfg_key in (
            "model_dtype",
            "mixed_precision",
            "zero_stage",
            "param_offload",
            "optimizer_offload",
        ):
            lines.append(f"  - {cfg_key}: {summary.config.get(cfg_key)}")
        if summary.final_validation:
            lines.append("- Final validation metrics:")
            for key, value in summary.final_validation.items():
                lines.append(f"  - {key}: {format_metric(value)}")
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_markdown_with_images(summaries: List[RunSummary], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# DeepSpeed PPO 四种设置对比（每个设置一张图）\n")
    for summary in summaries:
        slug = re.sub(r"[^0-9a-zA-Z_-]+", "_", summary.label)
        img_name = f"metrics_run_{slug}.png"
        lines.append(f"## {summary.label}")
        lines.append(f"![{summary.label}]({img_name})")
        lines.append("- 配置要点：")
        for cfg_key in (
            "trainer.nnodes",
            "trainer.n_gpus_per_node",
            "derived_data_parallel_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "ppo_mini_batch_size",
            "ppo_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
        ):
            lines.append(f"  - {cfg_key}: {summary.config.get(cfg_key)}")
        reward = (
            summary.final_metrics.get("reward")
            or summary.final_validation.get("val-core/openai/gsm8k/reward/mean@1")
        )
        lines.append(f"- 最终 reward（最后一步）: {format_metric(reward)}")
        lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def format_metric(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def plot_metrics(summaries: List[RunSummary], output_path: Path) -> None:
    metrics_to_plot = [
        ("reward", "Validation Reward"),
        ("actor_entropy", "Actor Entropy"),
        ("actor_pg_loss", "Actor Policy Loss"),
        ("critic_vf_loss", "Critic VF Loss"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metrics_to_plot):
        for summary in summaries:
            steps = []
            values = []
            for step_metrics in sorted(summary.steps, key=lambda s: s.step):
                metric_value = step_metrics.raw.get(metric_key)
                if metric_value is None or math.isnan(metric_value):
                    continue
                steps.append(step_metrics.step)
                values.append(metric_value)
            if steps and values:
                ax.plot(steps, values, marker="o", label=summary.label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()

    axes[0].set_ylabel("Reward")
    axes[2].set_ylabel("Loss / Entropy")

    fig.suptitle("DeepSpeed PPO Training Metrics")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_single(summaries: List[RunSummary], metric_key: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for summary in summaries:
        steps, values = [], []
        for sm in sorted(summary.steps, key=lambda s: s.step):
            val = sm.raw.get(metric_key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            steps.append(sm.step)
            values.append(val)
        if steps and values:
            ax.plot(steps, values, marker="o", label=summary.label)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_run_grid(summaries: List[RunSummary], output_dir: Path) -> None:
    """For each run, draw a 2x2 grid (same layout as metrics_progress.png)
    containing only that case's curves.
    """
    metric_spec = [
        ("reward", "Validation Reward"),
        ("actor_entropy", "Actor Entropy"),
        ("actor_pg_loss", "Actor Policy Loss"),
        ("critic_vf_loss", "Critic VF Loss"),
    ]

    for summary in summaries:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", constrained_layout=True)
        axes = axes.flatten()

        for ax, (metric_key, title) in zip(axes, metric_spec):
            steps, values = [], []
            for sm in sorted(summary.steps, key=lambda s: s.step):
                val = sm.raw.get(metric_key)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                steps.append(sm.step)
                values.append(val)
            if steps and values:
                ax.plot(steps, values, marker="o", label=summary.label)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend()

        axes[0].set_ylabel("Reward")
        axes[2].set_ylabel("Loss / Entropy")

        # Title with basic setting info
        title_cfg = []
        dp = summary.config.get("derived_data_parallel_size")
        tp = summary.config.get("tensor_model_parallel_size")
        if dp:
            title_cfg.append(f"DP={dp}")
        if tp:
            title_cfg.append(f"TP={tp}")
        nn = summary.config.get("trainer.nnodes")
        ng = summary.config.get("trainer.n_gpus_per_node")
        if nn and ng:
            title_cfg.append(f"{nn}x{ng} GPUs")
        fig.suptitle(f"{summary.label}  (" + ", ".join(title_cfg) + ")")

        out_path = output_dir / f"metrics_progress_{_slugify(summary.label)}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _slugify(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_-]+", "_", text)


def plot_per_run_combined(summaries: List[RunSummary], output_dir: Path) -> None:
    for summary in summaries:
        series = {
            "reward": [],
            "actor_entropy": [],
            "actor_pg_loss": [],
            "critic_vf_loss": [],
        }
        steps: List[int] = []
        for sm in sorted(summary.steps, key=lambda s: s.step):
            has_any = False
            for k in series.keys():
                v = sm.raw.get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    has_any = True
            if not has_any:
                continue
            steps.append(sm.step)
            for k in series.keys():
                series[k].append(sm.raw.get(k))

        if not steps:
            continue

        fig, ax_left = plt.subplots(figsize=(8, 5))
        ax_right = ax_left.twinx()

        lns = []
        labels = []

        colors = {
            "reward": "#1f77b4",
            "actor_entropy": "#2ca02c",
            "actor_pg_loss": "#d62728",
            "critic_vf_loss": "#ff7f0e",
        }
        styles = {
            "reward": "-",
            "actor_entropy": "-.",
            "actor_pg_loss": ":",
            "critic_vf_loss": "--",
        }

        def _plot(ax, key: str):
            vals = series[key]
            valid_x, valid_y = [], []
            for x, y in zip(steps, vals):
                if y is None or (isinstance(y, float) and math.isnan(y)):
                    continue
                valid_x.append(x)
                valid_y.append(y)
            if not valid_x:
                return None
            return ax.plot(valid_x, valid_y, linestyle=styles[key], color=colors[key], linewidth=1.8, label=key)[0]

        for key in ("reward", "actor_entropy"):
            line = _plot(ax_left, key)
            if line is not None:
                lns.append(line)
                labels.append(key)

        for key in ("actor_pg_loss", "critic_vf_loss"):
            line = _plot(ax_right, key)
            if line is not None:
                lns.append(line)
                labels.append(key)

        ax_left.set_xlabel("Step")
        ax_left.set_ylabel("Reward / Entropy")
        ax_right.set_ylabel("Loss")

        title_cfg = []
        tp = summary.config.get("tensor_model_parallel_size")
        dp = summary.config.get("derived_data_parallel_size")
        if dp:
            title_cfg.append(f"DP={dp}")
        if tp:
            title_cfg.append(f"TP={tp}")
        nn = summary.config.get("trainer.nnodes")
        ng = summary.config.get("trainer.n_gpus_per_node")
        if nn and ng:
            title_cfg.append(f"{nn}x{ng} GPUs")
        title_tail = "  (" + ", ".join(title_cfg) + ")" if title_cfg else ""
        ax_left.set_title(f"{summary.label}{title_tail}")

        if lns:
            ax_left.legend(lns, labels, loc="best")

        ax_left.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        out_path = output_dir / f"metrics_run_{_slugify(summary.label)}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def print_console_summary(summaries: List[RunSummary]) -> None:
    header = f"{'Run':<15} {'Step':>6} {'Reward':>10} {'Entropy':>10} {'PG Loss':>10} {'VF Loss':>10} {'Thpt':>10}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        final_metrics = summary.final_metrics
        print(
            f"{summary.label:<15} "
            f"{(summary.final_step or 0):>6} "
            f"{format_metric(final_metrics.get('reward')):>10} "
            f"{format_metric(final_metrics.get('actor_entropy')):>10} "
            f"{format_metric(final_metrics.get('actor_pg_loss')):>10} "
            f"{format_metric(final_metrics.get('critic_vf_loss')):>10} "
            f"{format_metric(final_metrics.get('throughput')):>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DeepSpeed PPO benchmark runs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing log files.",
    )
    parser.add_argument("--log", action="append", help="Override log mapping using LABEL:/path/to/log")
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional output path for the plot image.")
    parser.add_argument("--csv-path", type=Path, default=None, help="Optional CSV summary output path.")
    parser.add_argument("--json-path", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--markdown-path", type=Path, default=None, help="Optional Markdown summary output path.")
    parser.add_argument(
        "--skip-split", action="store_true", help="Skip single-metric comparison plots to save time."
    )
    parser.add_argument(
        "--skip-per-run", action="store_true", help="Skip per-setting plots (metrics_run_* and grid variants)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()

    run_map = merge_run_map(results_dir, args.log)
    ensure_run_files(run_map)

    summaries = summarize_runs(run_map)
    print_console_summary(summaries)

    plot_path = args.plot_path if args.plot_path is not None else results_dir / "metrics_progress.png"
    plot_metrics(summaries, plot_path)

    csv_path = args.csv_path if args.csv_path is not None else results_dir / "metrics_summary.csv"
    write_csv(summaries, csv_path)

    json_path = args.json_path if args.json_path is not None else results_dir / "metrics_summary.json"
    write_json(summaries, json_path)

    markdown_path = args.markdown_path if args.markdown_path is not None else results_dir / "metrics_summary.md"
    write_markdown(summaries, markdown_path)

    if not args.skip_split:
        # split per-metric comparison plots
        for key, title, outp in [
            ("reward", "Validation Reward", results_dir / "metrics_reward.png"),
            ("actor_entropy", "Actor Entropy", results_dir / "metrics_actor_entropy.png"),
            ("actor_pg_loss", "Actor Policy Loss", results_dir / "metrics_actor_pg_loss.png"),
            ("critic_vf_loss", "Critic VF Loss", results_dir / "metrics_critic_vf_loss.png"),
        ]:
            plot_metric_single(summaries, key, title, outp)

    if not args.skip_per_run:
        # per-setting combined chart (single axes with four curves)
        plot_per_run_combined(summaries, results_dir)
        # per-setting 2x2 grid (same layout as global metrics_progress.png)
        plot_per_run_grid(summaries, results_dir)

    # Chinese case report with images
    write_markdown_with_images(summaries, results_dir / "case_report_cn.md")


if __name__ == "__main__":
    main()
