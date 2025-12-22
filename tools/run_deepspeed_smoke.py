"""Run DeepSpeed PPO smoke tests (LoRA/offload/SGLang) with small settings.

Examples:
    python3 tools/run_deepspeed_smoke.py --mode lora_dp2_sp1
    python3 tools/run_deepspeed_smoke.py --mode offload_dp2_sp2 --steps 30

Modes:
    - lora_dp2_sp1  (2 GPUs, DP only)
    - lora_dp1_sp2  (2 GPUs, SP only)
    - lora_dp2_sp2  (4 GPUs, DP+SP)
    - offload_dp2_sp2 (4 GPUs, ZeRO-3 param+optimizer offload)
    - sglang_dp2_sp2 (4 GPUs, SGLang rollout; skips if sglang not installed)

The script will:
    * Prepare a tiny parquet dataset (samples existing GSM8K if available, otherwise synthetic).
    * Build hydra overrides for DeepSpeed worker + async rollout.
    * Run ~N training steps (default 30) and emit logs proving LoRA/offload flags are active.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from tools.make_tiny_parquet import DEFAULT_ROWS, prepare_tiny_dataset

DEFAULT_MODEL = os.environ.get("VERL_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

SMOKE_MODES: Dict[str, Dict[str, object]] = {
    "lora_dp2_sp1": {"dp": 2, "sp": 1, "lora": True, "offload": False, "sglang": False},
    "lora_dp1_sp2": {"dp": 1, "sp": 2, "lora": True, "offload": False, "sglang": False},
    "lora_dp2_sp2": {"dp": 2, "sp": 2, "lora": True, "offload": False, "sglang": False},
    "offload_dp2_sp2": {"dp": 2, "sp": 2, "lora": False, "offload": True, "sglang": False},
    "sglang_dp2_sp2": {"dp": 2, "sp": 2, "lora": False, "offload": False, "sglang": True},
}


def _check_cuda(required_gpus: int, env: dict) -> tuple[bool, str]:
    """Return (ok, message) after ensuring CUDA_VISIBLE_DEVICES is set."""
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(required_gpus))
    else:
        visible_count = len([x for x in visible.split(",") if x.strip() != ""])
        if visible_count < required_gpus:
            return False, f"CUDA_VISIBLE_DEVICES only exposes {visible_count} GPU(s) (< {required_gpus})."

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is expected to be present in smoke envs
        return False, f"torch unavailable: {exc}"

    if not torch.cuda.is_available():
        return False, "CUDA not available."

    count = torch.cuda.device_count()
    if count < required_gpus:
        return False, f"Need {required_gpus} GPU(s), only {count} detected."

    return True, f"{count} GPU(s) visible (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})."


def _check_sglang() -> tuple[bool, str]:
    try:
        import sglang  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"sglang not importable: {exc}. Install with `pip install -e '.[sglang]'`."
    return True, f"sglang=={getattr(sglang, '__version__', 'unknown')}"


def _check_vllm() -> tuple[bool, str]:
    try:
        import vllm  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"vLLM not importable: {exc}. Install with `pip install -e .` or `pip install vllm`."
    return True, f"vLLM=={getattr(vllm, '__version__', 'unknown')}"


def _build_overrides(
    mode: str, train_path: Path, val_path: Path, model_path: str, steps: int
) -> tuple[List[str], int]:
    cfg = SMOKE_MODES[mode]
    dp = int(cfg["dp"])
    sp = int(cfg["sp"])
    world = dp * sp
    is_sglang = bool(cfg["sglang"])
    overrides: List[str] = [
        "actor@actor_rollout_ref.actor=deepspeed_actor",
        "critic=deepspeed_critic",
        "actor_rollout_ref.actor.strategy=deepspeed",
        f"trainer.n_gpus_per_node={world}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={sp}",
        f"critic.ulysses_sequence_parallel_size={sp}",
        f"actor_rollout_ref.actor.deepspeed_config.ulysses_sequence_parallel_size={sp}",
        f"critic.deepspeed_config.ulysses_sequence_parallel_size={sp}",
        "actor_rollout_ref.actor.deepspeed_config.model_dtype=bf16",
        "critic.deepspeed_config.model_dtype=bf16",
        f"actor_rollout_ref.rollout.name={'sglang' if is_sglang else 'vllm'}",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"trainer.total_training_steps={steps}",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.log_val_generations=0",
        "trainer.logger=[console]",
        f"trainer.experiment_name=deepspeed_smoke_{mode}",
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        "data.max_prompt_length=256",
        "data.max_response_length=128",
        "data.train_batch_size=64",
        "data.val_batch_size=16",
        "data.shuffle=True",
        "data.validation_shuffle=False",
        f"actor_rollout_ref.model.path={model_path}",
        "+actor_rollout_ref.model.override_config.attn_implementation=eager",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.ignore_eos=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "critic.ppo_mini_batch_size=4",
        "critic.ppo_micro_batch_size_per_gpu=1",
        f"critic.model.path={model_path}",
        f"critic.model.tokenizer_path={model_path}",
        "algorithm.use_kl_in_reward=False",
        "algorithm.use_pf_ppo=False",
    ]

    if cfg["lora"]:
        overrides.extend(
            [
                "actor_rollout_ref.model.lora_rank=8",
                "actor_rollout_ref.model.lora_alpha=16",
                "actor_rollout_ref.model.target_modules=all-linear",
            ]
        )

    if cfg["offload"]:
        overrides.extend(
            [
                "actor_rollout_ref.actor.zero_stage=3",
                "critic.zero_stage=3",
                "actor_rollout_ref.actor.deepspeed_config.param_offload=True",
                "actor_rollout_ref.actor.deepspeed_config.optimizer_offload=True",
                "critic.deepspeed_config.param_offload=True",
                "critic.deepspeed_config.optimizer_offload=True",
            ]
        )

    return overrides, world


def _run(cmd: List[str], log_path: Path, env: dict) -> tuple[int, dict]:
    """Execute command, tee logs, and capture simple health markers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = {"saw_lora_arg": False, "saw_offload": False, "saw_nan": False}
    nan_re = re.compile(r"\bnan\b", re.IGNORECASE)

    print(f"\n>>> Running: {' '.join(shlex.quote(x) for x in cmd)}")
    print(f">>> Log: {log_path}")
    print(f">>> CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
    if env.get("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
        print(">>> SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=true")
    if env.get("WANDB_MODE"):
        print(f">>> WANDB_MODE={env['WANDB_MODE']}")
    with log_path.open("w") as handle:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
            lower = line.lower()
            if "enable_lora" in lower or "max_lora_rank" in lower:
                flags["saw_lora_arg"] = True
            if "[ds actor] offload config" in lower or "[ds critic] offload config" in lower:
                flags["saw_offload"] = True
            if nan_re.search(lower):
                flags["saw_nan"] = True
        proc.wait()
        return proc.returncode, flags


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed PPO smoke runner")
    parser.add_argument("--mode", required=True, choices=SMOKE_MODES.keys(), help="Which smoke scenario to run.")
    parser.add_argument("--steps", type=int, default=30, help="Training steps to run.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model name or local path (override with VERL_SMOKE_MODEL env as well).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "data" / "gsm8k",
        help="Where to place the tiny parquet data.",
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Rows to keep in the tiny dataset.")
    args = parser.parse_args()

    mode_cfg = SMOKE_MODES[args.mode]
    if not args.model:
        print(f"Model argument empty, falling back to default: {DEFAULT_MODEL}")
        args.model = DEFAULT_MODEL
    world = int(mode_cfg["dp"]) * int(mode_cfg["sp"])
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Keep GPUs visible even if Ray actors don't explicitly request them.
    env.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if mode_cfg["offload"]:
        env.setdefault("VERL_DISABLE_DEEPSPEED_CPU_ADAM", "1")
        env.setdefault("DS_BUILD_CPU_ADAM", "0")
        env.setdefault("DS_BUILD_ADAM", "0")
        env.setdefault("DS_BUILD_FUSED_ADAM", "0")
        env.setdefault("DS_SKIP_CUDA_CHECK", "1")
    if mode_cfg["sglang"]:
        ok_sgl, msg = _check_sglang()
        if not ok_sgl:
            print(f"SKIP: {msg}")
            sys.exit(0)
        env.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        print(f"SGLang check: {msg}")
    else:
        ok_vllm, msg = _check_vllm()
        if not ok_vllm:
            print(f"SKIP: {msg}")
            sys.exit(0)
        print(f"vLLM check: {msg}")

    # Prepare dataset
    train_path, val_path, note = prepare_tiny_dataset(args.data_dir, rows=args.rows)
    print(f"Data ready: train={train_path}, val={val_path} ({note})")

    ok_cuda, cuda_msg = _check_cuda(world, env)
    if not ok_cuda:
        print(f"SKIP: {cuda_msg}")
        sys.exit(0)
    print(f"CUDA check: {cuda_msg}")

    overrides, _ = _build_overrides(
        mode=args.mode, train_path=train_path, val_path=val_path, model_path=args.model, steps=args.steps
    )

    log_dir = Path("outputs") / "deepspeed_smoke"
    log_path = log_dir / f"{args.mode}.log"

    cmd = ["python3", "-m", "verl.trainer.main_ppo", "-cn", "ppo_trainer"] + overrides
    exit_code, flags = _run(cmd, log_path, env)

    if exit_code != 0:
        print(f"FAIL: process exited with code {exit_code}. Logs: {log_path}")
        sys.exit(exit_code)

    if mode_cfg["lora"] and not flags["saw_lora_arg"]:
        print(f"FAIL: LoRA mode but did not see enable_lora/max_lora_rank in logs. See {log_path}")
        sys.exit(1)

    if mode_cfg["offload"] and not flags["saw_offload"]:
        print(f"FAIL: Offload mode but did not see DeepSpeed offload config logged. See {log_path}")
        sys.exit(1)

    if flags["saw_nan"]:
        print(f"FAIL: Detected 'nan' in logs. See {log_path}")
        sys.exit(1)

    print(f"SUCCESS: mode={args.mode} finished. Logs: {log_path}")


if __name__ == "__main__":
    main()
