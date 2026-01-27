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

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import torch
from verl.workers.config.actor import DistillationConfig  
from verl.base_config import BaseConfig

class Stage(Enum):
    """Stages for on-policy distillation training."""

    OLD_LOG_PROB = "old_log_prob"
    REF_LOG_PROB = "ref_log_prob"
    ACTOR_UPDATE = "actor_update"

def get_topk_keys(stage: str | Stage) -> tuple[str, str]:
    """Get the TensorDict keys for storing top-k log probabilities and indices for a given stage."""
    if isinstance(stage, Stage):
        stage = stage.value
    return f"{stage}_topk_log_probs", f"{stage}_topk_indices"

def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled

@dataclass
class DistillationLossInputs(BaseConfig):
    """Storage class for distillation loss inputs."""

    student_log_probs: Optional[torch.Tensor] = None
    teacher_log_probs: Optional[torch.Tensor] = None
    student_topk_logprobs: Optional[torch.Tensor] = None
    teacher_topk_logprobs: Optional[torch.Tensor] = None
    student_topk_indices: Optional[torch.Tensor] = None
    teacher_topk_indices: Optional[torch.Tensor] = None