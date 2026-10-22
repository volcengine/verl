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

import math
import random
from collections.abc import Callable
from functools import partial
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl.utils.import_utils import load_extern_object


def is_teacher_rollout_enabled(config: DictConfig) -> bool:
    """Whether training may use a user-selected teacher generation source."""
    return bool(OmegaConf.select(config, "distillation.rollout_source_selector.path"))


def build_rollout_source_selector(config: DictConfig) -> Callable[..., Literal["student", "teacher"]] | None:
    """Load a synchronous user function once; policies own their selection and reproducibility.

    The function receives keyword arguments ``batch``, ``global_step``, and ``seed``,
    plus configured kwargs. It returns one source for the entire submission.
    No selector preserves the existing student generation path.
    """
    if not is_teacher_rollout_enabled(config):
        return None
    selector = config.distillation.rollout_source_selector
    function = load_extern_object(selector.path, selector.name)
    if not callable(function):
        raise TypeError("distillation.rollout_source_selector must refer to a callable.")
    kwargs = OmegaConf.to_container(selector.get("kwargs", OmegaConf.create({})), resolve=True)
    return partial(function, **kwargs)


def select_source(
    *, batch: TensorDict, global_step: int, seed: int, teacher_probability: float = 1.0
) -> Literal["student", "teacher"]:
    """Built-in fixed-probability selector; draws from (seed, global_step) without touching global RNG state."""
    if not math.isfinite(teacher_probability) or not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must be finite and in [0, 1].")
    if teacher_probability == 0.0:
        return "student"
    if teacher_probability == 1.0:
        return "teacher"
    draw = random.Random(f"teacher-rollout:{seed}:{global_step}").random()
    return "teacher" if draw < teacher_probability else "student"
