"""DeepSpeed checkpoint manager.

Reference implementation style aligned with `FSDPCheckpointManager` while keeping
DeepSpeed's native save/load helpers. Provides:

    - Directory layout: <root>/step_<global_step>/
    - Automatic pruning (keep newest `max_ckpt_to_keep`)
    - Latest checkpoint auto-discovery when `step` is None
    - CPU offload friendly: temporarily loads params to GPU if ZeRO offload enabled

The implementation intentionally stays lightweight and does not subclass
`BaseCheckpointManager` yet (DeepSpeed's internal checkpoint API bundles model,
optimizer, and lr scheduler state automatically). If convergence with the FSDP
manager API becomes desirable, this class can be adapted to inherit from the
base without changing its external engine integration.
"""

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from typing import Any

import torch
import torch.distributed

from verl.utils.device import get_device_name

try:  # DeepSpeed imports are optional
    from deepspeed import DeepSpeedEngine as _DSEngine  # noqa: F401
except Exception:  # pragma: no cover - env without deepspeed
    _DSEngine = object  # type: ignore

# Try multiple import locations for DS checkpoint helpers.
# Fall back to VERL's wrappers (engine.save_checkpoint / load_checkpoint) if unavailable.
_SAVE_FN = None
_LOAD_FN = None

try:
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import (  # DS >= 0.13
        load_deepspeed_checkpoint as _LOAD_FN,  # type: ignore
        save_deepspeed_checkpoint as _SAVE_FN,  # type: ignore
    )
except Exception:  # pragma: no cover - compatibility path
    try:
        from deepspeed.runtime.checkpoint_engine.torch import (  # older DS
            load_deepspeed_checkpoint as _LOAD_FN,  # type: ignore
            save_deepspeed_checkpoint as _SAVE_FN,  # type: ignore
        )
    except Exception:  # pragma: no cover - last resort to internal wrappers
        try:
            # Use VERL's deepspeed_utils wrappers as a stable fallback
            from verl.utils.deepspeed_utils import (  # type: ignore
                load_deepspeed_checkpoint as _LOAD_FN,
                save_deepspeed_checkpoint as _SAVE_FN,
            )
        except Exception:
            _SAVE_FN = None
            _LOAD_FN = None


def save_deepspeed_checkpoint(*, engine, save_dir, client_state=None, tag=None, **kwargs):  # type: ignore
    if _SAVE_FN is not None:
        return _SAVE_FN(engine=engine, save_dir=save_dir, client_state=client_state, tag=tag)
    # Fallback to engine API directly
    if hasattr(engine, "save_checkpoint"):
        return engine.save_checkpoint(save_dir, client_state=client_state, tag=tag)
    raise RuntimeError("DeepSpeed checkpoint save helper not available and engine.save_checkpoint missing")


def load_deepspeed_checkpoint(
    *,
    engine,
    load_dir,
    tag=None,
    load_module_strict: bool = True,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
    **kwargs,
):  # type: ignore
    if _LOAD_FN is not None:
        return _LOAD_FN(
            engine=engine,
            load_dir=load_dir,
            tag=tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
    # Fallback to engine API directly
    if hasattr(engine, "load_checkpoint"):
        ret = engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        # DeepSpeed returns (load_path, client_state) in many versions
        if isinstance(ret, tuple) and len(ret) >= 2:
            return ret[1]
        return {}
    raise RuntimeError("DeepSpeed checkpoint load helper not available and engine.load_checkpoint missing")


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Optional offload helpers (no-op if absent)
try:
    from verl.workers.engine.deepspeed.offload_utils import (  # hypothetical existing util
        load_deepspeed_model_to_gpu,
        offload_deepspeed_model_to_cpu,
    )
except Exception:  # pragma: no cover

    def load_deepspeed_model_to_gpu(engine):  # type: ignore
        return

    def offload_deepspeed_model_to_cpu(engine):  # type: ignore
        return


class DeepSpeedCheckpointManager:
    """Lightweight checkpoint manager for DeepSpeedEngine.

    Notes:
        - Uses DeepSpeed's native save/load APIs (model + optimizer + lr scheduler bundled)
        - Implements pruning & latest discovery similar to the FSDP manager design
        - Safe to use with ZeRO param offload (temporarily loads params to GPU during I/O)
    """

    CKPT_DIR_PREFIX = "step_"

    def __init__(self, engine: Any):  # engine expected to expose .engine, .rank, and _is_offload_param flag
        self.engine = engine
        self.rank = getattr(engine, "rank", 0)
        self.device_name = get_device_name()

    # ----- internal helpers -----
    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _parse_step(self, dirname: str):
        if not dirname.startswith(self.CKPT_DIR_PREFIX):
            return None
        m = re.match(rf"{self.CKPT_DIR_PREFIX}(\d+)$", dirname)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None

    def _list_ckpts(self, root: str):
        if not os.path.isdir(root):
            return []
        items = []
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                step = self._parse_step(d)
                if step is not None:
                    items.append((step, full))
        return sorted(items, key=lambda x: x[0])

    def _latest(self, root: str):
        ckpts = self._list_ckpts(root)
        return ckpts[-1] if ckpts else None

    def _prune(self, root: str, max_keep: int | None):
        if not max_keep or max_keep <= 0:
            return
        ckpts = self._list_ckpts(root)
        if len(ckpts) <= max_keep:
            return
        stale = ckpts[:-max_keep]
        for step, path in stale:
            if self.rank == 0:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"[DeepSpeedCheckpointManager] Pruned checkpoint step={step} path={path}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[DeepSpeedCheckpointManager] Failed to prune {path}: {e}")
        if torch.distributed.is_initialized():  # best-effort sync
            try:  # pragma: no cover
                torch.distributed.barrier()
            except Exception:  # noqa: BLE001
                pass

    # ----- public API -----
    def save(self, root: str, global_step: int, hdfs_path: str | None = None, max_ckpt_to_keep: int | None = None):
        self._ensure_dir(root)
        target_dir = os.path.join(root, f"{self.CKPT_DIR_PREFIX}{global_step}")
        self._ensure_dir(target_dir)

        if getattr(self.engine, "_is_offload_param", False):
            load_deepspeed_model_to_gpu(self.engine.engine)

        client_state: dict[str, Any] = {
            "global_step": global_step,
            "saved_time": datetime.utcnow().isoformat(),
            "hdfs_path": hdfs_path,
        }

        # We already materialize a unique directory per step, so no need to pass a DeepSpeed tag that
        # would create a nested step_<n>/step_<n>/ hierarchy. Using tag=None keeps files flat.
        save_deepspeed_checkpoint(
            engine=self.engine.engine,
            save_dir=target_dir,
            client_state=client_state,
            tag=None,
        )

        if torch.distributed.is_initialized():  # pragma: no cover - distributed only
            try:
                torch.distributed.barrier()
            except Exception:  # noqa: BLE001
                pass

        if getattr(self.engine, "_is_offload_param", False):
            offload_deepspeed_model_to_cpu(self.engine.engine)

        self._prune(root, max_ckpt_to_keep)
        if self.rank == 0:
            logger.info(f"[DeepSpeedCheckpointManager] Saved checkpoint step={global_step} dir={target_dir}")
        return target_dir

    def load(
        self,
        root: str,
        step: int | None = None,
        hdfs_path: str | None = None,  # kept for symmetry / potential remote sync
        del_local_after_load: bool = False,
    ):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Checkpoint root not found: {root}")
        if step is None:
            latest = self._latest(root)
            if latest is None:
                raise FileNotFoundError(f"No checkpoints under {root}")
            step, path = latest
        else:
            path = os.path.join(root, f"{self.CKPT_DIR_PREFIX}{step}")
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Checkpoint step {step} missing at {path}")

        if getattr(self.engine, "_is_offload_param", False):
            load_deepspeed_model_to_gpu(self.engine.engine)

        client_state = load_deepspeed_checkpoint(
            engine=self.engine.engine,
            load_dir=path,
            tag=None,  # latest inside that directory
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )

        if torch.distributed.is_initialized():  # pragma: no cover
            try:
                torch.distributed.barrier()
            except Exception:  # noqa: BLE001
                pass

        if getattr(self.engine, "_is_offload_param", False):
            offload_deepspeed_model_to_cpu(self.engine.engine)

        if del_local_after_load and self.rank == 0:
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"[DeepSpeedCheckpointManager] Removed checkpoint after load: {path}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to remove {path}: {e}")

        if self.rank == 0:
            logger.info(f"[DeepSpeedCheckpointManager] Loaded checkpoint step={step} from {path}")
        return client_state or {}
