# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 Benchmax Team
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

from typing import Any, Optional
from uuid import uuid4

from benchmax.envs.base_env import BaseEnv, ToolDefinition

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.schemas import OpenAIFunctionSchema


class BenchmaxToolAdapter(BaseTool):
    """
    Wrap one benchmax tool behind the BaseTool interface expected by the
    verl evaluator/trainer stack.

    - `instance_id` ↔ `rollout_id`: we simply pass the value straight through.
    - Tool-level Rewards are left as zero;
    """

    def __init__(self, benchmax_env: BaseEnv, tool_def: ToolDefinition):
        self._benchmax_env = benchmax_env
        self._tool_def = tool_def
        self.initialized_requests = set([])
        tool_schema = OpenAIFunctionToolSchema(
            type="object",
            function=OpenAIFunctionSchema(
                name=self._tool_def.name,
                description=self._tool_def.description or "",
                parameters=self._tool_def.input_schema or {},
            ),
        )
        super().__init__(config={}, tool_schema=tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Mint an ID."""
        instance_id = instance_id or str(uuid4())
        if instance_id not in self.initialized_requests:
            self.initialized_requests.add(instance_id)
            self._benchmax_env.init_rollout(instance_id, **kwargs)
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **_) -> tuple[str, float, dict]:
        """
        Forward the call to the underlying benchmax env tool, injecting `rollout_id`
        so everything downstream can treat it as part of the correct rollout.
        """
        response = self._benchmax_env.run_tool(instance_id, self._tool_def.name, **parameters)
        return str(response), 0.0, {}  # (tool_response, step_reward, metrics)

    async def calc_reward(self, *args, **kwargs) -> float:
        """No per-step reward in this generic adapter."""
        return 0.0

    async def release(self, instance_id: str, **_) -> None:
        """Release the tool instance."""
        if instance_id in self.initialized_requests:
            self._benchmax_env.cleanup_rollout(instance_id)
            self.initialized_requests.remove(instance_id)

    def get_rollout_workspace(self, instance_id: str) -> str:
        """
        Get the workspace directory for the given instance_id.
        This is useful for tools that need to access files or directories.
        """
        return self._benchmax_env.get_rollout_workspace(instance_id)


def benchmax_env_to_tool_list(benchmax_env: BaseEnv) -> list[BaseTool]:
    """
    Convert all tools registered inside a benchmax `BaseEnv` into `BaseTool` instances.
    """
    return [BenchmaxToolAdapter(benchmax_env, tool_def) for tool_def in benchmax_env.list_tools()]
