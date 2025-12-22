"""Minimal demo to verify DeepSpeed ZeRO-3 state dict gathering for vLLM weight push.

The script:
  - Builds a tiny torch model (two Linear layers).
  - Wraps it with DeepSpeed ZeRO stage 3 (optional offload flags).
  - Runs a dummy forward pass.
  - Gathers full parameters via `_zero3_consolidated_state_dict` or `GatheredParameters`.
  - Compares against the original full state dict (CPU) to ensure shapes/values match.

Usage:
  python3 tools/demo_deepspeed_zero3_gather.py --offload-optimizer --offload-param

This is CPU-only friendly but will use CUDA if available. World size is 1 for simplicity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Tuple

import deepspeed
import torch
from deepspeed.runtime.zero import GatheredParameters


def _sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu()
    if arr.dtype == torch.bfloat16:
        arr = arr.float()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()[:16]


def _hash_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, str]:
    return {k: _sha256_tensor(v) for k, v in sd.items()}


def _compare_state_dicts(
    base: Dict[str, torch.Tensor], gathered: Dict[str, torch.Tensor], atol: float = 1e-3
) -> Tuple[int, float]:
    mismatched = 0
    max_diff = 0.0
    for k, v in base.items():
        if k not in gathered:
            mismatched += 1
            continue
        gv = gathered[k]
        if v.shape != gv.shape:
            mismatched += 1
            continue
        diff = (v.detach().cpu().float() - gv.detach().cpu().float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > atol:
            mismatched += 1
    return mismatched, max_diff


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 gather demo")
    parser.add_argument("--offload-optimizer", action="store_true", help="Enable optimizer_offload=True")
    parser.add_argument("--offload-param", action="store_true", help="Enable param_offload=True")
    args = parser.parse_args()

    torch.manual_seed(0)

    # Tiny model
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(8, 16)
            self.lin2 = torch.nn.Linear(16, 4)

        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))

    base_model = TinyModel()
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    # DeepSpeed config (ZeRO-3)
    ds_config = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True} if args.offload_param else None,
            "offload_optimizer": {"device": "cpu", "pin_memory": True} if args.offload_optimizer else None,
        },
        "bf16": {"enabled": torch.cuda.is_available()},
        "steps_per_print": 1,
    }
    # Remove empty offload dicts when flag is False
    if not args.offload_param:
        ds_config["zero_optimization"].pop("offload_param", None)
    if not args.offload_optimizer:
        ds_config["zero_optimization"].pop("offload_optimizer", None)

    print("DeepSpeed config:", json.dumps(ds_config, indent=2))

    # Ensure distributed init (single process)
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.distributed.init_process_group(
                backend="nccl", init_method="tcp://127.0.0.1:23456", rank=0, world_size=1
            )
        else:
            torch.distributed.init_process_group(
                backend="gloo", init_method="tcp://127.0.0.1:23456", rank=0, world_size=1
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    engine, _, _, _ = deepspeed.initialize(
        model=base_model, model_parameters=base_model.parameters(), config=ds_config
    )

    # Dummy forward
    dummy = torch.randn(1, 8, device=device, dtype=next(base_model.parameters()).dtype)
    with torch.no_grad():
        out = engine(dummy)
    print("Forward ok, output shape:", out.shape)

    # Gather weights
    if hasattr(engine, "_zero3_consolidated_state_dict"):
        gathered = engine._zero3_consolidated_state_dict()  # type: ignore[attr-defined]
        print("Used _zero3_consolidated_state_dict")
    else:
        print("Fallback to GatheredParameters")
        with GatheredParameters(engine.module.parameters(), modifier_rank=None):
            gathered = {k: v.detach().cpu() for k, v in engine.module.state_dict().items()}

    # Hash and compare
    mismatched, max_diff = _compare_state_dicts(base_state, gathered)
    print(f"Gathered {len(gathered)} params, mismatched={mismatched}, max_abs_diff={max_diff:.4e}")
    if mismatched:
        for k in list(base_state)[:5]:
            if k not in gathered:
                print(f"  missing in gathered: {k}")
            elif base_state[k].shape != gathered[k].shape:
                print(f"  shape mismatch: {k} base={base_state[k].shape} gathered={gathered[k].shape}")
            else:
                diff = (
                    base_state[k].detach().cpu().float() - gathered[k].detach().cpu().float()
                ).abs().max()
                print(f"  diff: {k} max_abs={diff.item():.4e}")
    else:
        print("All parameters match within tolerance.")


if __name__ == "__main__":
    main()
