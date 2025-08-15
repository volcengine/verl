# Copyright 2025 Meituan Ltd. and/or its affiliates
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
from dataclasses import dataclass
from typing import Any


# Calculate the number of samples needed
def calculate_one_step_size(minimal_bsz, ppo_mini_batch_size):
    return minimal_bsz * ppo_mini_batch_size


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information (preserved from _prepare_generate_batch)
    original_batch_dict: dict[str, Any]

    # AgentLoopOutput from generation
    agent_loop_output: Any  # AgentLoopOutput

    # Metadata
    sample_id: str
    epoch: int
    rollout_n_index: int  # Index within the rollout.n repetitions (0, 1, ..., n-1)
    original_sample_index: int  # Index of the original sample before repetition

    # Processing metadata
    processing_time: float
    generation_timestamp: float
    param_version: int

    _gen_data: Any
