# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import logging
import os
import threading
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4
from contextlib import ExitStack
import time
import ray
import ray.actor

from verl.utils.reward_score import prime_code
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case, call_sandbox_api

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2
    


@ray.remote(concurrency_groups={"acquire": 1,"release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._lock = threading.Lock()

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        while(1):
            with self._lock:
                if self.current_count < self.rate_limit:
                    self.current_count += 1
                    return True
            #TODO: backoff
            time.sleep(1)

    @ray.method(concurrency_group="release")
    def release(self):
        with self._lock:
            self.current_count -=1

    def get_current_count(self):
            return self.current_count


@ray.remote
class ExecutionWorker:
    def __init__(self,enable_global_rate_limit=True,rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self,rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True
    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            return fn(*fn_args, **fn_kwargs)


def init_execution_pool(num_workers: int, worker_actor_cls: ray.actor.ActorClass,mode: PoolMode=PoolMode.ThreadMode):
    if mode == PoolMode.ThreadMode:
        return worker_actor_cls.options(max_concurrency=num_workers).remote()
    else:
        raise NotImplementedError
        return ray.util.ActorPool([worker_actor_cls.remote() for _ in range(num_workers)])

class SandboxFusionTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_code_result",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.execution_pool = init_execution_pool(num_workers=self.num_workers, worker_actor_cls=ExecutionWorker)
        self.sandbox_fusion_url = config.get("sandbox_fusion_url","")
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        print(f"self._instance_dict: {self._instance_dict}, prime_tools create are called")
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code,instance_id,code)
        # penalty for non improved answer submission
        # tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        # print(f"self._instance_dict: {self._instance_dict}, prime_tools execute are called")
        self._instance_dict[instance_id]["reward"].append(result.strip())

        return result, result, {}

    def execute_code(self,instance_id,code):
        '''
            _process_single_case(
            case_index: int,
            stdin_data: Any,
            expected_output: Any,
            sandbox_fusion_url: str,
            generation: str,
            timeout: int,
            language: str
        )
        '''
        # TODO make this into asyncio format: 
        result_status, metadata  = _process_single_case(0, None, None,self.sandbox_fusion_url, code, 30, "python")
        # we should always expect this since we don't have correct answer
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
            print(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
            return actual_output
        else:
            return "no stdout here"


    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        # this code only called as a cumulation reward, so we return the sandbox result
        # only for unit test to do any kind of verification
        # print(f"self._instance_dict: {self._instance_dict}, prime_tools calc_reward are called")
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
