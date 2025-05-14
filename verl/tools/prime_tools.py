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

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import prime_code
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case, call_sandbox_api

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PrimeTool(BaseTool):
    """A demo tool for calculating the reward of prime.

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
                "description": "A tool for calculating the reward of prime",
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
        self.sandbox_fusion_url = config.get("sandbox_fusion_url","URL_ADDRESSxxxx.apigateway-cn-beijing.volceapi.com/run_code")

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

        result = await self.execute_code(instance_id,code)
        # penalty for non improved answer submission
        # tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        print(f"self._instance_dict: {self._instance_dict}, prime_tools execute are called")
        self._instance_dict[instance_id]["reward"].append(result.strip())

        return result, result, {}

    async def execute_code(self,instance_id,code):
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
            print("actual_output from sandbox fusion: ",actual_output)
            return actual_output
        else:
            return "no stdout here"


    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        # this code only called as a cumulation reward, so we return the sandbox result
        # only for unit test to do any kind of verification
        print(f"self._instance_dict: {self._instance_dict}, prime_tools calc_reward are called")
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
