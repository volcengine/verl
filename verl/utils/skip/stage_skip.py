# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training-loop stage skips: reward, old_log_prob, ref_log_prob, values, actor_update, weight_sync.

Stage roles wrap trainer methods shaped like ``method(self, batch, ...) -> metrics``
that write their outputs as columns of ``batch`` (see verl-omni#716). The cache for
one (role, step) is the stage's declared batch columns plus the returned metrics,
stored under ``{dump_dir}/{exp}_{proj}/GBS{gbs}/{step}/{role}/``.

``empty`` is a debug-only no-op accepted by ``actor_update`` and ``weight_sync``:
the wrapped call does not run and ``empty_result`` is returned instead.
"""

import json
from pathlib import Path
from typing import Callable, ClassVar

import torch
from omegaconf import OmegaConf

from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip


class TrainerStageSkip(BaseSkip):
    """Base for training-loop stage roles.

    Subclasses declare ``role`` (registry name, used to namespace the dump dir),
    ``columns`` (batch fields the stage produces), ``support_actions`` and — for
    stages supporting ``empty`` — ``empty_result`` (never mutated, returned as-is).
    """

    support_online_step = False
    role: ClassVar[str] = "stage"
    columns: ClassVar[tuple[str, ...]] = ()
    empty_result: ClassVar[object] = None
    stage_file_name = "stage.pt"
    meta_name = "meta.json"
    print_mark = "[TrainerStageSkip()] "

    def __init__(self, local_config, global_config):
        super().__init__(local_config, global_config)
        self.exp_name = global_config.trainer.get("experiment_name", "default_experiment_name")
        self.project_name = global_config.trainer.get("project_name", "default_project_name")
        gen_batch_size = OmegaConf.select(global_config, "data.gen_batch_size")
        if gen_batch_size is None:
            gen_batch_size = OmegaConf.select(global_config, "data.train_batch_size", default=0)
        self.gbs = int(gen_batch_size)

    def meet_precondition(self, step: int, func: Callable, *args, **kwargs) -> bool:
        if self.action == SkipAction.EMPTY:
            return True
        if self.action == SkipAction.CACHE:
            if not self._check_valid_stage_dir(self._get_stage_dump_dir(step)):
                print(
                    f"{self.print_mark}\033[33mNo dumped data found for stage {self.role} at step {step} "
                    f"from {self._get_project_dump_dir()}. "
                    f"The trainer will run the stage and dump the result.\033[0m",
                    flush=True,
                )
                return False
            return True
        if self.action == SkipAction.REPEAT:
            if self._find_latest_step(step) == -1:
                print(
                    f"{self.print_mark}\033[33mNo dumped data found for stage {self.role} "
                    f"from {self._get_project_dump_dir()}. "
                    f"The trainer will run the stage and dump the result.\033[0m",
                    flush=True,
                )
                return False
            return True
        return False

    def warp_function(self, step: int, func: Callable, *args, **kwargs):
        """Inject cached columns into the batch and return the cached metrics."""
        if self.action == SkipAction.EMPTY:
            print(
                f"{self.print_mark}\033[33mSkip stage {self.role} at step {step} via empty action\033[0m",
                flush=True,
            )
            return self.empty_result
        if self.action == SkipAction.CACHE:
            load_step = step
        else:
            load_step = self._find_latest_step(step)
            if load_step == -1:
                raise RuntimeError(
                    f"{self.print_mark}repeat action expected dumped data for stage {self.role} "
                    f"at step {step}, but none was found under {self._get_project_dump_dir()}"
                )
        stage_dir = self._get_stage_dump_dir(load_step)
        payload = torch.load(stage_dir.joinpath(self.stage_file_name), weights_only=False)
        batch = self._get_stage_batch(args, kwargs)
        if batch is not None:
            # tensor columns live in ``batch.batch`` for DataProto (``batch`` itself for TensorDict)
            tensors = getattr(batch, "batch", batch)
            non_tensor = getattr(batch, "non_tensor_batch", None)
            for name, value in payload["non_tensor_columns"].items():
                if non_tensor is not None:
                    non_tensor[name] = value
                else:
                    tensors[name] = value
            for name, value in payload["tensor_columns"].items():
                tensors[name] = value
        print(
            f"{self.print_mark}\033[33mLoad stage {self.role} result at step {load_step} "
            f"(request step {step}) from {stage_dir}\033[0m",
            flush=True,
        )
        return payload["metrics"]

    def prepare_data(self, step: int, result, *args, **kwargs):
        if self.action == SkipAction.EMPTY:
            return
        stage_dir = self._get_stage_dump_dir(step)
        try:
            batch = self._get_stage_batch(args, kwargs)
            tensor_columns, non_tensor_columns = {}, {}
            if batch is not None:
                tensors = getattr(batch, "batch", batch)
                non_tensor = getattr(batch, "non_tensor_batch", None)
                for name in self.columns:
                    if non_tensor is not None and name in non_tensor:
                        non_tensor_columns[name] = non_tensor[name]
                    else:
                        tensor_columns[name] = tensors[name]
            payload = {"tensor_columns": tensor_columns, "non_tensor_columns": non_tensor_columns, "metrics": result}
            stage_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, stage_dir.joinpath(self.stage_file_name))
            stage_dir.joinpath(self.meta_name).write_text(json.dumps({"global_steps": step}))
            print(
                f"{self.print_mark}\033[33mDump stage {self.role} result at step {step} to {stage_dir}\033[0m",
                flush=True,
            )
        except Exception as e:
            print(
                f"{self.print_mark}\033[31mFailed to dump stage {self.role} result at step {step} "
                f"to {stage_dir}: {e}\033[0m",
                flush=True,
            )

    @staticmethod
    def _get_stage_batch(args, kwargs):
        """Resolve the stage batch from decorated ``method(self, batch, ...)`` calls."""
        batch = kwargs.get("batch")
        if batch is not None:
            return batch
        if len(args) > 1:
            return args[1]
        return None

    def _get_project_dump_dir(self) -> Path:
        dumped_dir = Path(self.dump_dir).expanduser().resolve()
        return dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}", f"GBS{self.gbs}").absolute()

    def _get_stage_dump_dir(self, step) -> Path:
        return self._get_project_dump_dir().joinpath(f"{step}", self.role).absolute()

    def _check_valid_stage_dir(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        stage_path = path.joinpath(self.stage_file_name)
        meta_path = path.joinpath(self.meta_name)
        return stage_path.exists() and stage_path.is_file() and meta_path.exists() and meta_path.is_file()

    def _get_available_steps(self) -> list[int]:
        result: list[int] = []
        project_dir = self._get_project_dump_dir()
        if not project_dir.is_dir():
            return result
        for child in project_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                step = int(child.name)
            except ValueError:
                continue
            if not self._check_valid_stage_dir(child.joinpath(self.role)):
                continue
            result.append(step)
        return sorted(result)

    def _find_latest_step(self, step: int) -> int:
        """Prefer exact ready step, else max step < current, else min step > current; -1 if none."""
        if self._check_valid_stage_dir(self._get_stage_dump_dir(step)):
            return step
        available = self._get_available_steps()
        if not available:
            return -1
        smaller_steps = [this_step for this_step in available if this_step < step]
        if smaller_steps:
            return smaller_steps[-1]
        larger_steps = [this_step for this_step in available if this_step > step]
        if larger_steps:
            return larger_steps[0]
        return -1


@register_skip("reward")
class RewardSkip(TrainerStageSkip):
    """Skip colocated reward scoring; cache holds the reward columns of the batch."""

    role = "reward"
    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    columns = ("rm_scores", "token_level_rewards")


@register_skip("old_log_prob")
class OldLogProbSkip(TrainerStageSkip):
    """Skip the old-log-prob recompute pass."""

    role = "old_log_prob"
    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    columns = ("old_log_probs",)


@register_skip("ref_log_prob")
class RefLogProbSkip(TrainerStageSkip):
    """Skip the reference-policy log-prob recompute pass."""

    role = "ref_log_prob"
    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    columns = ("ref_log_prob",)


@register_skip("values")
class ValuesSkip(TrainerStageSkip):
    """Skip the critic value inference pass."""

    role = "values"
    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    columns = ("values",)


@register_skip("actor_update")
class ActorUpdateSkip(TrainerStageSkip):
    """Skip the actor update. ``cache`` replays stored update metrics; ``empty`` is a no-op."""

    role = "actor_update"
    support_actions = [SkipAction.CACHE, SkipAction.EMPTY]
    columns = ()
    empty_result: ClassVar[dict] = {}  # noqa: RUF012 — shared immutable placeholder, never mutated


@register_skip("weight_sync")
class WeightSyncSkip(TrainerStageSkip):
    """Skip syncing weights to the rollout engine — debug only, stalls rollout progress.

    Requires ``skip.weight_sync.ack_debug=True`` (enforced by
    ``SkipManager._assert_stage_conflicts``).
    """

    role = "weight_sync"
    support_actions = [SkipAction.CACHE, SkipAction.EMPTY]
    columns = ()
    empty_result = None
