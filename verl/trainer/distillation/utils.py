# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from tensordict import TensorDict
import torch
from verl.trainer.distillation.fsdp import utils
from verl.trainer.distillation.megatron import utils
from verl.trainer.distillation.common import Stage, DistillationLossInputs
from verl.workers.config.actor import DistillationConfig


def prepare_distillation_inputs(
    log_prob: torch.Tensor, data: TensorDict, model_output: dict[str, torch.Tensor], config: DistillationConfig
) -> DistillationLossInputs:
    match config.strategy:
        case "fsdp":
            return utils.prepare_distillation_inputs(log_prob=log_prob, data=data, model_output=model_output, config=config)
        case "megatron":
            return utils.prepare_distillation_inputs(log_prob=log_prob, data=data, model_output=model_output, config=config)
        case _:
            raise ValueError(f"Unsupported distillation strategy: {config.strategy}")

def extract_distillation_inputs(
    stage: Stage, output: TensorDict, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    match config.strategy:
        case "fsdp":
            return utils.extract_distillation_inputs(stage=stage, output=output, config=config)
        case "megatron":
            return utils.extract_distillation_inputs(stage=stage, output=output, config=config)
        case _:
            raise ValueError(f"Unsupported distillation strategy: {config.strategy}")
        