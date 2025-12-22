"""Create a tiny parquet dataset compatible with VERL PPO smoke tests.

The helper will try to:
1) Sample a handful of rows from an existing GSM8K parquet (if present).
2) Otherwise, synthesize a minimal dataset with the expected RLHF fields.

The resulting train/val parquets live under the target output directory.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_ROWS = 64


def _synthetic_rows(num_rows: int) -> list[dict]:
    """Generate a tiny synthetic dataset with the required columns."""
    templates: list[dict] = [
        {
            "data_source": "synthetic_math",
            "ability": "math",
            "prompt": [{"role": "user", "content": "Add 1 and 2. Output the sum only."}],
            "reward_model": {"style": "rule", "ground_truth": "3"},
        },
        {
            "data_source": "synthetic_math",
            "ability": "math",
            "prompt": [{"role": "user", "content": "What is 7 minus 5? Return the number only."}],
            "reward_model": {"style": "rule", "ground_truth": "2"},
        },
        {
            "data_source": "synthetic_trivia",
            "ability": "qa",
            "prompt": [{"role": "user", "content": "Capital of France?"}],
            "reward_model": {"style": "rule", "ground_truth": "Paris"},
        },
        {
            "data_source": "synthetic_reason",
            "ability": "reasoning",
            "prompt": [{"role": "user", "content": "If X=2 and Y=3, what is X*Y+1?"}],
            "reward_model": {"style": "rule", "ground_truth": "7"},
        },
    ]

    rows: list[dict] = []
    for idx in range(num_rows):
        base = copy.deepcopy(templates[idx % len(templates)])
        base["extra_info"] = {"split": "train", "index": idx}
        base["uid"] = f"synthetic-{idx}"
        rows.append(base)
    return rows


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _candidate_sources(source: Path | None, base_dir: Path) -> Iterable[Path]:
    """Yield existing parquet sources to sample from."""
    if source is not None and source.is_file():
        yield source
    default_source = base_dir / "train.parquet"
    if default_source.is_file():
        yield default_source


def _maybe_build_from_gsm8k(base_dir: Path) -> Path | None:
    """Best-effort: run the GSM8K preprocessing script to produce a parquet."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "data_preprocess" / "gsm8k.py"
    if not script.is_file():
        return None

    try:
        print(f"Attempting to build GSM8K parquet via {script} ...")
        cmd = [sys.executable, str(script), "--local_save_dir", str(base_dir)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        candidate = base_dir / "train.parquet"
        return candidate if candidate.is_file() else None
    except Exception as exc:  # pragma: no cover - offline fallback path
        print(f"Warning: failed to run gsm8k.py ({exc}); falling back to synthetic data.")
        return None


def prepare_tiny_dataset(output_dir: Path, rows: int = DEFAULT_ROWS, source: Path | None = None):
    """
    Create tiny train/val parquet files.

    Returns:
        (train_path, val_path, note)
    """
    output_dir = output_dir.expanduser()
    source = source.expanduser() if source is not None else None

    train_path = output_dir / "tiny_train.parquet"
    val_path = output_dir / "tiny_val.parquet"

    for candidate in _candidate_sources(source, output_dir):
        df = pd.read_parquet(candidate)
        if rows > 0:
            df = df.head(rows)
        note = f"sampled {len(df)} rows from {candidate}"
        _write(df, train_path)
        _write(df.head(max(1, min(len(df), rows // 4 or 1))), val_path)
        return train_path, val_path, note

    built = _maybe_build_from_gsm8k(output_dir)
    if built is not None and built.is_file():
        df = pd.read_parquet(built)
        if rows > 0:
            df = df.head(rows)
        note = f"built GSM8K parquet via {built}"
        _write(df, train_path)
        _write(df.head(max(1, min(len(df), rows // 4 or 1))), val_path)
        return train_path, val_path, note

    rows = max(rows, 1)
    df = pd.DataFrame(_synthetic_rows(rows))
    note = f"generated synthetic dataset with {rows} rows"
    _write(df, train_path)
    _write(df.head(max(1, rows // 4)), val_path)
    return train_path, val_path, note


def main():
    parser = argparse.ArgumentParser(description="Create a tiny VERL-compatible parquet dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "data" / "gsm8k",
        help="Where to write tiny_train.parquet / tiny_val.parquet",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional existing parquet to sample from (defaults to <output-dir>/train.parquet if omitted).",
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="How many rows to keep.")
    args = parser.parse_args()

    train_path, val_path, note = prepare_tiny_dataset(args.output_dir, rows=args.rows, source=args.source)
    print(f"Wrote train parquet: {train_path}")
    print(f"Wrote val parquet:   {val_path}")
    print(f"Note: {note}")


if __name__ == "__main__":
    main()
