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
import importlib
import json
import sys
from typing import Any, List, Optional, Tuple
from uuid import uuid4

from omegaconf import OmegaConf

from .schemas import OpenAIFunctionToolSchema


class BaseTool:
    """Base class for tools.

    A tool should support the following methods:

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name
        print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        return "Updated the tool state.", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        pass


def initialize_tools_from_config(tools_config_file) -> List[BaseTool]:
    """Initialize tools from config file.

    Args:
        tools_config_file: The config file of the tools.

    Returns:
        A list of tools.
    """
    tools_config = OmegaConf.load(tools_config_file)

    tool_list = []
    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name
        module_name, class_name = cls_name.rsplit(".", 1)

        if module_name not in sys.modules:
            spec = importlib.util.find_spec(module_name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_name]

        tool_cls = getattr(module, class_name)

        if tool_config.get("tool_schema", None) is None:
            tool_schema = None
        else:
            tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
            tool_schema = OpenAIFunctionToolSchema.parse_obj(tool_schema_dict)

        tool = tool_cls(config=OmegaConf.to_container(tool_config.config, resolve=True), tool_schema=tool_schema)
        tool_list.append(tool)

    return tool_list
