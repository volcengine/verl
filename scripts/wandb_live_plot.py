#!/usr/bin/env python3
"""Generate a live-updating W&B plot PNG from offline run history."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402

try:
    import wandb  # noqa: E402
except Exception:  # pragma: no cover - optional dependency for API usage
    wandb = None

METRICS = [
    "critic/rewards/mean",
    "val-aux/openai/gsm8k/reward/mean@1",
    "val-core/openai/gsm8k/acc/mean@1",
    "actor/kl_loss",
]


def _latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.glob("offline-run-*") if p.is_dir()]
    if not candidates:
        candidates = [p for p in runs_root.glob("run-*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RUN_NAME_RE = re.compile(r"wandb:\s+Syncing run\s+(.+)$")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _parse_step_line(line: str) -> tuple[int, dict] | None:
    if "step:" not in line:
        return None
    clean = _strip_ansi(line)
    idx = clean.find("step:")
    if idx == -1:
        return None
    segment = clean[idx:].strip()
    parts = segment.split(" - ")
    if not parts:
        return None
    step_part = parts[0]
    try:
        step = int(step_part.split("step:")[1].strip())
    except Exception:
        return None
    metrics: dict[str, float] = {}
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    if not metrics:
        return None
    return step, metrics


def _read_metrics_from_log(
    log_path: Path,
    offset: int,
    cache: dict[int, dict[str, float]],
    meta: dict[str, str],
) -> int:
    if not log_path.exists():
        return offset
    try:
        if log_path.stat().st_size < offset:
            offset = 0
            cache.clear()
    except OSError:
        return offset
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        for line in handle:
            clean = _strip_ansi(line).strip()
            if "wandb: Syncing run" in clean:
                match = _RUN_NAME_RE.search(clean)
                if match:
                    meta["run_name"] = match.group(1).strip()
            parsed = _parse_step_line(line)
            if parsed is None:
                continue
            step, metrics = parsed
            existing = cache.get(step, {})
            existing.update(metrics)
            cache[step] = existing
        return handle.tell()


def _moving_average(values: Iterable[float], window: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr
    # Expanding moving average until we have a full window.
    out = np.empty_like(arr, dtype=float)
    cumsum = np.cumsum(arr)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def _series(records: list[dict], metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for rec in records:
        if metric not in rec:
            continue
        value = rec.get(metric)
        if value is None:
            continue
        step = rec.get("_step")
        if step is None:
            step = rec.get("step")
        if step is None:
            step = len(xs)
        try:
            xs.append(int(step))
        except (TypeError, ValueError):
            continue
        ys.append(float(value))
    if not xs:
        return [], []
    order = np.argsort(xs)
    xs_sorted = [xs[i] for i in order]
    ys_sorted = [ys[i] for i in order]
    return xs_sorted, ys_sorted


def _series_from_cache(cache: dict[int, dict[str, float]], metric: str) -> tuple[list[int], list[float]]:
    if not cache:
        return [], []
    steps = sorted(cache.keys())
    xs: list[int] = []
    ys: list[float] = []
    for step in steps:
        value = cache[step].get(metric)
        if value is None:
            continue
        xs.append(step)
        ys.append(float(value))
    return xs, ys


def _write_placeholder(out_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_plot(
    records: list[dict],
    out_path: Path,
    window: int,
    cache: dict | None = None,
    title: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=120)
    axes = axes.flatten()
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    for ax, metric in zip(axes, METRICS, strict=False):
        if records:
            xs, ys = _series(records, metric)
        elif cache is not None:
            xs, ys = _series_from_cache(cache, metric)
        else:
            xs, ys = [], []
        ax.set_title(metric)
        ax.grid(True, alpha=0.2)
        max_step = max(xs) if xs else None
        if max_step is not None and max_step < 50:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.set_xlabel("step")
        if not xs:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center", fontsize=10)
            continue
        ax.plot(xs, ys, color="#7aa6f9", alpha=0.5, linewidth=1.5, label="raw")
        smoothed = _moving_average(ys, window)
        if smoothed.size > 0:
            ax.plot(xs, smoothed, color="#f28e2b", linewidth=2, label=f"ma({window})")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    fig.savefig(out_path)
    plt.close(fig)


def _fetch_wandb_records(
    entity: str | None,
    project: str | None,
    run_id: str | None,
    timeout: float,
) -> tuple[list[dict], str | None]:
    if wandb is None or entity is None or project is None:
        return [], None
    try:
        api = wandb.Api(timeout=timeout)
        if run_id:
            run = api.run(f"{entity}/{project}/{run_id}")
        else:
            runs = api.runs(f"{entity}/{project}")
            if not runs:
                return [], None
            run = max(runs, key=lambda r: getattr(r, "updated_at", "") or "")
        records = list(run.scan_history(keys=["_step", "step"] + METRICS))
        max_step = None
        for rec in records:
            step = rec.get("_step", rec.get("step"))
            if step is None:
                continue
            max_step = max(step, max_step) if max_step is not None else step
        title = None
        if max_step is not None:
            run_name = getattr(run, "name", run_id or "run")
            run_state = getattr(run, "state", "running")
            title = f"{run_name} | state={run_state} | step={int(max_step)}"
        return records, title
    except Exception:
        return [], None


def main() -> None:
    wandb_dir = Path(os.environ.get("WANDB_DIR", "/home/ubuntu/verl/wandb"))
    runs_root = Path(os.environ.get("WANDB_RUNS_ROOT", wandb_dir / "wandb"))
    out_png = Path(os.environ.get("WANDB_LIVE_PLOT", "/home/ubuntu/verl/wandb_live_plot.png"))
    out_html = Path(os.environ.get("WANDB_LIVE_HTML", "/home/ubuntu/verl/wandb_live_plot.html"))
    poll_seconds = float(os.environ.get("WANDB_POLL_SECONDS", "60"))
    window = int(os.environ.get("WANDB_SMOOTH_WINDOW", "15"))
    log_path = Path(os.environ.get("WANDB_LOG_PATH", "/home/ubuntu/verl/atropos_grpo_small.log"))
    refresh_seconds = int(os.environ.get("WANDB_REFRESH_SECONDS", str(int(poll_seconds))))
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    run_id = os.environ.get("WANDB_RUN_ID")
    api_only = os.environ.get("WANDB_API_ONLY", "1") == "1"
    api_timeout = float(os.environ.get("WANDB_API_TIMEOUT", "60"))
    api_enabled = os.environ.get("WANDB_SKIP_API", "0") != "1"

    # Write a simple auto-refresh page once.
    if not out_html.exists():
        out_html.write_text(
            """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="60" />
    <title>W&B Live Plot</title>
    <style>body{font-family:Arial, sans-serif; margin:20px;} img{max-width:100%;}</style>
  </head>
  <body>
    <h2>W&B Live Plot</h2>
    <img src="/wandb_live_plot.png" alt="wandb live plot" />
  </body>
</html>
""",
            encoding="utf-8",
        )
    # Ensure refresh interval matches poll setting.
    html_text = out_html.read_text(encoding="utf-8")
    html_text = html_text.replace('http-equiv="refresh" content="60"', f'http-equiv="refresh" content="{refresh_seconds}"')
    out_html.write_text(html_text, encoding="utf-8")

    last_history_size = -1
    log_offset = 0
    log_cache: dict[int, dict[str, float]] = {}
    log_meta: dict[str, str] = {}
    last_api_step: int | None = None
    while True:
        api_records: list[dict] = []
        api_title: str | None = None
        if api_enabled:
            api_records, api_title = _fetch_wandb_records(entity, project, run_id, api_timeout)

        run_dir = _latest_run_dir(runs_root)
        history_path = run_dir / "files" / "wandb-history.jsonl" if run_dir else None

        records: list[dict] = []
        if not api_only and history_path and history_path.exists():
            try:
                current_size = history_path.stat().st_size
            except OSError:
                current_size = -1
            if current_size != last_history_size:
                records = _read_history(history_path)
                last_history_size = current_size
        if not api_only and not records:
            log_offset = _read_metrics_from_log(log_path, log_offset, log_cache, log_meta)

        if api_records:
            current_api_step = None
            for rec in api_records:
                step = rec.get("_step", rec.get("step"))
                if step is None:
                    continue
                current_api_step = max(step, current_api_step) if current_api_step is not None else step
            if current_api_step is not None and current_api_step == last_api_step:
                time.sleep(poll_seconds)
                continue
            last_api_step = current_api_step
            _render_plot(api_records, out_png, window, title=api_title)
        elif records:
            _render_plot(records, out_png, window)
        elif log_cache:
            max_step = max(log_cache.keys()) if log_cache else None
            run_name = log_meta.get("run_name", "run")
            title = None
            if max_step is not None:
                title = f"{run_name} | state=running | step={int(max_step)}"
            _render_plot([], out_png, window, cache=log_cache, title=title)
        else:
            if run_dir is None:
                message = f"Waiting for W&B runs in {runs_root}..."
            else:
                message = f"Waiting for metrics in {log_path}..."
            _write_placeholder(out_png, message)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
