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

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar, Union
from uuid import uuid4

import ray
import ray.actor
import asyncio

from verl.tools.utils.mcp_clients.McpClientManager import ClientManager

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class MCPSearchTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = config.get("timeout", 30)
        # TODO(alec henx): create a global client manager to manage the rate limit, client and pool
        logger.info(f"Initialized MCPSearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        timeout = self.timeout
        if self.name == "" or self.name is None or parameters is None:
            error_msg = "Error: 'parameters' is missing or empty."
            logger.error(f"[MCPTool] {error_msg} Received tool name: {self.name}, parameters: {parameters}")
            return json.dumps({"result": error_msg}), 0.0, {}

        try:
            session = await ClientManager.get_client_from_tool(self.name)
            if session is None:
                logger.error(f"session is `None` for {self.name}")
                return json.dumps({"result": f"session is None for {self.name}"}), 0.0, {}

            call_tool_result = await session.call_tool(self.name, parameters, timeout)
            logger.debug(f"Search result for instance {instance_id} with tool {self.name}: {call_tool_result.content}")
            tools_content = [
                part.text
                for part in filter(lambda x: x.type == "text", call_tool_result.content)
            ]
            result_text = " ".join(tools_content)

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())
            return result_text, 0.0, {}

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[MCPSearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
