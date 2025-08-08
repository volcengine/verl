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
from dataclasses import dataclass
from typing import Optional

import torch

from recipe.reorder_rollout.chat_scheduler.utils import ActorMeta
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


@dataclass
class RolloutReq:
    # sample_id works for n-samples, n replicated requests share the same sample_id
    sample_id: Optional[str]

    # maybe we can count the requeue times
    generation: int = 0


@dataclass
class RolloutResp:
    request: RolloutReq
    exception: Optional[Exception] = None


@dataclass
class CallsReq:
    rollout_resp: RolloutResp
    actor_meta: ActorMeta


@dataclass
class ReduceResp:
    agent_loop_output_list: list[AgentLoopOutput]
    response_mask: torch.Tensor
    response_ids: torch.Tensor
    response_attention_mask: torch.Tensor
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
