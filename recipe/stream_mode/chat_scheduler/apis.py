from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from recipe.stream_mode.chat_scheduler.utils import ActorMeta
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


@dataclass
class RolloutReq:
    # sample_id works for n-samples, n replicated requests share the same sample_id
    sample_id: Optional[str]
    model_name: str
    messages: List[Dict[str, str]]
    sampling_params: Dict[str, Any]
    agent_name: np.ndarray
    # maybe we can count the requeue times
    generation: int = 0
    # this works for tool-calling, indicate for one-multi-turns request
    verl_session_id: Optional[str] = None


@dataclass
class RolloutResp:
    request: RolloutReq
    exception: Optional[Exception] = None
    req_id: str = None
    agent_loop_output: AgentLoopOutput = None


@dataclass
class CallsReq:
    rollout_resp: RolloutResp
    actor_meta: ActorMeta


@dataclass
class ReduceResp:
    raw_prompt: Optional[np.ndarray]
    agent_loop_output_list: List[AgentLoopOutput]
