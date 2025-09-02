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

import asyncio
import logging
import multiprocessing as mp
import os
import time
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
# from verl.workers.config import 
from verl.workers.rollout.async_server import TokenOutput
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
from verl.workers.reward_model import BasePPORewardModel


class SGLangReward(BasePPORewardModel):
    def __init__(
        self,
        actor_module: str,
        config: RolloutConfig,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        pass